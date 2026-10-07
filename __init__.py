"""jiabaili 评分包。角色权重数据来源：异环工坊（小程序）。

计算标准（整套满分 350 = 空幕 35 格 × 10 分）：
- 核心件（卡带）：(主词条权重 × 50 + 副词条权重 / 理论满权重 × 100) × 品质
- 盘件（驱动块）：(副词条权重 / 理论满权重 × 面积 × 10) × 品质
- 理论满权重 = 该角色副词条权重最高的 4 个之和

权重数据策略：
- 包内不预置任何权重数据；首次成功请求接口后，把权重缓存到本地 data/weights_cache.json。
- 后台任务每天 00:00 / 06:00 / 12:00 / 18:00 请求接口并刷新该缓存文件。
- 每次评分优先实时请求接口，失败则降级用缓存文件计算；两者都不可用则报「评分数据缺失」。

高亮规则：副词条只亮权重最高的 4 个，主词条只亮权重最高的（并列都亮）。
核心件（卡带）与盘件（驱动块）主词条数据可能缺失（服务端只回 id 缺 name/value），
此时按「魂属性异能伤害增强」判定高亮与主词条权重。
核心件与盘件均不挂评级徽章。

总评分评级（350 满分固定分档）：ACE≥280 / SSS≥260 / SS≥240 / S≥220 /
A+≥200 / A≥180 / B≥160 / C≥140 / D<140。
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections.abc import Sequence

import httpx

from gsuid_core.logger import logger

from ...contract import GradeSpec, BaseScorer, ScorerMeta
from ...registry import register_scorer
from ....utils.sdk.tajiduo_model import CharacterDetail, CharacterProperty, CharacterSuitItem
from ....utils.resource.RESOURCE_PATH import SCORING_PATH

_DATA = Path(__file__).parent / "data"
_ASSETS = Path(__file__).parent / "assets"
# 接口成功后落盘的本地缓存文件（首次评分时建立，每天定时刷新）
_CACHE_PATH = _DATA / "weights_cache.json"

# 权重数据源：异环工坊开放接口。
# 策略：首次安装不带任何权重数据；首次评分运行请求接口，成功即把权重缓存到
# weights_cache.json；往后每天 06:00 / 12:00 / 18:00 / 00:00 自动请求接口刷新该缓存。
# 每次评分优先实时请求接口取最新权重，失败时降级到该缓存文件做计算。
_API_URL = "https://yh.zzzmap.com/api/open/game-character/weight-configs"
_API_TIMEOUT = 8.0
# 每天自动刷新缓存的整点（本地时区）
_REFRESH_HOURS = (0, 6, 12, 18)

# 实时接口的内存缓存：避免批量评分时几十次角色各发一次请求
_REMOTE_WEIGHTS: dict | None = None
_FETCH_LOCK: asyncio.Lock | None = None
# 每天定时刷新任务
_DAILY_TASK: asyncio.Task | None = None

# 整套满分：空幕 35 格 × 10 分
FULL_SET_SCORE = 350.0
# 核心件单件理论满分：主词条 50 + 副词条 100
CORE_MAIN_SCORE = 50.0
CORE_SUB_SCORE = 100.0
CORE_MAX_SCORE = CORE_MAIN_SCORE + CORE_SUB_SCORE
# 驱动块每格满分
PIE_SCORE_PER_AREA = 10.0

# 总评级阈值：按总分（350 满分）的固定分档划分
# 注意：原表 C 档写作 ≥160（与 B 同分），D < 140，140~160 未覆盖；
# 这里按单调分档的意图取 C ≥ 140，使 140~160 落入 C、<140 落入 D。
FULL_GRADES = (
    (280.0, "ACE"),
    (260.0, "SSS"),
    (240.0, "SS"),
    (220.0, "S"),
    (200.0, "A+"),
    (180.0, "A"),
    (160.0, "B"),
    (140.0, "C"),
    (0.0, "D"),
)

# 单件评级阈值：按得分 / 该件理论满分的比例划分
PIECE_GRADES = (
    (0.95, "ACE"),
    (0.85, "SSS"),
    (0.75, "SS"),
    (0.65, "S"),
    (0.55, "A+"),
    (0.45, "A"),
    (0.30, "B"),
    (0.15, "C"),
    (0.0, "D"),
)

# 塔吉多装备数据用双 p（damageuppsychebase），工坊权重接口用单 p（damageupsychebase），
# 两者是同一属性「魂属性异能伤害增强」，按游戏侧拼写统一。
_PROP_ID_ALIASES = {"damageupsychebase": "damageuppsychebase"}

# 盘件（驱动块）与核心件（卡带）主词条服务端可能只回 `{id}` 缺 `name`/`value`
# （见 tajiduo_model.CharacterProperty docstring），无法从 name 反查词条 id，
# 默认按「魂属性异能伤害增强」判定高亮与主词条权重。
# 塔吉多侧拼写为 damageuppsychebase（双 p），与工坊单 p 别名见 _PROP_ID_ALIASES。
_DEFAULT_MAIN_PROP_ID = "damageuppsychebase"


def _canon_prop_id(prop_id: str) -> str:
    pid = (prop_id or "").lower()
    return _PROP_ID_ALIASES.get(pid, pid)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EquipmentView:
    item_id: str
    display: str
    grade: str | None
    unlocked_subs: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Result:
    score: float
    display: str
    grade: str
    equipment: tuple[_EquipmentView, ...]
    main_weights: dict[str, float] = field(default_factory=dict)
    sub_weights: dict[str, float] = field(default_factory=dict)
    top_main_ids: frozenset[str] = frozenset()
    top_sub_ids: frozenset[str] = frozenset()

    def is_role_prop_effective(self, prop: CharacterProperty) -> bool:
        # 角色面板：与装备词条高亮同一口径 —— 主词条只亮权重最高、副词条只亮权重最高的 4 个。
        # 属性 id 在角色面板可能与工坊权重 key 不同名（同一属性多种写法），
        # 故先按 id 精确命中；不命中再经 attributes.json 的 name -> ids 反查（同 yuye 做法）。
        pid = _canon_prop_id(prop.id)
        if not prop.name.strip() and not prop.value.strip():
            return pid == _DEFAULT_MAIN_PROP_ID
        if pid in self.top_main_ids or pid in self.top_sub_ids:
            return True
        return any(
            i in self.top_main_ids or i in self.top_sub_ids
            for i in _attr_name_ids().get(prop.name, ())
        )

    def is_main_prop_counted(self, prop: CharacterProperty) -> bool:
        # 装备主词条：只有「权重最高」的主词条高亮（核心件、盘件同一口径）。
        # 主词条若缺 name/value（数据缺失），默认按「魂属性异能伤害增强」判定。
        pid = _canon_prop_id(prop.id)
        if not prop.name.strip() and not prop.value.strip():
            return pid == _DEFAULT_MAIN_PROP_ID
        return pid in self.top_main_ids

    def is_sub_prop_recommended(self, prop: CharacterProperty) -> bool:
        # 装备副词条：只有「权重最高」的副词条高亮。
        return _canon_prop_id(prop.id) in self.top_sub_ids

    def highlight_color(self, prop: CharacterProperty, locked: bool) -> tuple[int, int, int] | None:
        """HighlightPalette 协议：高亮词条渲染为金色；未解锁副词条用暗金。"""
        if locked:
            return (180, 150, 50)
        return (255, 200, 64)


def _normalize_weights(raw: dict) -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    result: dict[str, dict[str, dict[str, float] | frozenset[str]]] = {}
    for char_id, data in raw.items():
        result[str(char_id)] = {
            "main": {_canon_prop_id(attr): float(weight) for attr, weight in (data.get("main") or {}).items()},
            "sub": {_canon_prop_id(attr): float(weight) for attr, weight in (data.get("sub") or {}).items()},
            "highlight": frozenset(_canon_prop_id(attr) for attr in (data.get("highlight") or [])),
        }
    return result


@lru_cache(maxsize=1)
def _cached_weights() -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    """本地兜底权重：首次接口成功后落盘的 data/weights_cache.json（每天 4 个整点定时刷新）。"""
    if not _CACHE_PATH.exists():
        raise ValueError(f"评分数据缺失: {_CACHE_PATH}（接口暂不可用，等待首次接口成功后缓存）")
    raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    return _normalize_weights(raw)


def _resolve_main_prop_id(prop: CharacterProperty) -> str:
    """主词条 id：数据缺失（缺 name/value）时默认按「魂属性异能伤害增强」。
    仅用于评分时取主词条权重；高亮判定仍以实际词条 id 为准。"""
    pid = _canon_prop_id(prop.id)
    if not prop.name.strip() and not prop.value.strip():
        return _DEFAULT_MAIN_PROP_ID
    return pid


def _weights() -> dict[str, dict[str, dict[str, float] | frozenset[str]]]:
    """生效权重：实时接口成功优先，否则降级到本地缓存文件。"""
    if _REMOTE_WEIGHTS is not None:
        return _REMOTE_WEIGHTS
    return _cached_weights()


def _parse_api_raw(payload: dict) -> dict[str, dict]:
    """异环工坊权重接口响应 -> 原始包内结构（highlight 为 list，可 JSON 序列化）。"""
    raw: dict[str, dict] = {}
    for char in payload.get("data") or []:
        char_id = str(char.get("itemId") or "").strip()
        if not char_id:
            continue
        wc = char.get("weightConfig") or {}
        weights = wc.get("weights") if isinstance(wc, dict) else (wc if isinstance(wc, list) else None)
        main: dict[str, float] = {}
        sub: dict[str, float] = {}
        highlight: list[str] = []
        for item in weights or []:
            key = _canon_prop_id(str(item.get("key") or ""))
            if not key:
                continue
            main_value = float(item.get("main_value") or 0)
            sub_value = float(item.get("value") or 0)
            if main_value > 0:
                main[key] = main_value
            if sub_value > 0:
                sub[key] = sub_value
            if item.get("highlight"):
                highlight.append(key)
        raw[char_id] = {
            "name": str(char.get("name") or ""),
            "main": main,
            "sub": sub,
            "highlight": highlight,
        }
    if not raw:
        raise ValueError("权重接口未解析到角色数据")
    return raw


def _fetch_lock() -> asyncio.Lock:
    global _FETCH_LOCK
    if _FETCH_LOCK is None:
        _FETCH_LOCK = asyncio.Lock()
    return _FETCH_LOCK


def _write_cache(raw: dict) -> None:
    """把接口解析出的原始权重结构（highlight 为 list，可 JSON 序列化）原子落盘。"""
    _DATA.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_CACHE_PATH)


async def _fetch_remote() -> dict | None:
    """实时请求异环工坊权重接口；成功返回归一化权重并同步落盘，失败返回 None。

    落盘存「归一化前的原始结构」（highlight 为 list），否则 frozenset 无法 JSON 序列化；
    内存缓存用归一化产物。失败写日志，便于定位为何没建立缓存。
    trust_env=False：不读系统代理等环境变量，避免 gsuid_core 运行环境下代理劫持握手。
    """
    try:
        async with httpx.AsyncClient(
            timeout=_API_TIMEOUT, follow_redirects=True, trust_env=False
        ) as client:
            resp = await client.get(_API_URL)
            resp.raise_for_status()
            payload = resp.json()
        raw = _parse_api_raw(payload)
        # 接口成功即落盘到本地缓存文件（首次评分即建立），保证兜底数据与线上同步
        _write_cache(raw)
        _cached_weights.cache_clear()
        logger.info(f"[jiabaili] 权重接口拉取成功，已缓存到 {_CACHE_PATH.name}")
        return _normalize_weights(raw)
    except Exception as error:
        logger.warning(f"[jiabaili] 权重接口请求失败，本次走本地缓存: {error!r}")
        return None


async def _refresh_weights(force: bool = False) -> None:
    """每次评分前调用：优先实时接口（成功更新内存缓存），失败保持现状走本地缓存。"""
    global _REMOTE_WEIGHTS
    if _REMOTE_WEIGHTS is not None and not force:
        return
    async with _fetch_lock():
        if _REMOTE_WEIGHTS is not None and not force:
            return
        weights = await _fetch_remote()
        if weights is not None:
            _REMOTE_WEIGHTS = weights
        # 失败则 _REMOTE_WEIGHTS 维持 None，_weights() 自动降级到缓存文件


def _seconds_until_next_refresh() -> float:
    """距下一个刷新整点（00:00 / 06:00 / 12:00 / 18:00，本地时区）的秒数。"""
    now = datetime.now()
    today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    candidates = [
        today_midnight + timedelta(days=day, hours=hour)
        for day in (0, 1)
        for hour in _REFRESH_HOURS
    ]
    nxt = min(t for t in candidates if t > now)
    return (nxt - now).total_seconds()


async def _daily_refresh_loop() -> None:
    """每天 00:00 / 06:00 / 12:00 / 18:00 定时请求接口并落盘到缓存文件。"""
    while True:
        await asyncio.sleep(_seconds_until_next_refresh())
        async with _fetch_lock():
            await _fetch_remote()


def _start_daily_task() -> None:
    global _DAILY_TASK
    if _DAILY_TASK is None or _DAILY_TASK.done():
        _DAILY_TASK = asyncio.ensure_future(_daily_refresh_loop())


def _stop_daily_task() -> None:
    global _DAILY_TASK
    if _DAILY_TASK is not None:
        _DAILY_TASK.cancel()
        _DAILY_TASK = None


@lru_cache(maxsize=1)
def _attr_names() -> dict[str, str]:
    path = SCORING_PATH / "attributes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(attr_id).lower(): str(info.get("name", "")) for attr_id, info in raw.items()}


@lru_cache(maxsize=1)
def _attr_name_ids() -> dict[str, tuple[str, ...]]:
    path = SCORING_PATH / "attributes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for attr_id, info in raw.items():
        name = str(info.get("name", ""))
        if name:
            result.setdefault(name, []).append(str(attr_id).lower())
    return {name: tuple(ids) for name, ids in result.items()}


def _piece_area(item: CharacterSuitItem) -> int | None:
    """驱动块面积：id 形如 cellN_xxx，N 即占格数；非 cell 开头为核心件（卡带）。"""
    if not item.id.startswith("cell"):
        return None
    try:
        return int(item.id.split("_", 1)[0][4:])
    except ValueError:
        return None


def _quality_factor(item_id: str) -> float:
    """品质系数：橙 1.0 / 紫 0.8 / 其余 0.6。"""
    if not item_id:
        return 0.6
    i = item_id.lower()
    if "orange" in i or "gold" in i:
        return 1.0
    if "purple" in i:
        return 0.8
    return 0.6


def _weight_for(prop_id: str, weights: dict[str, float]) -> float:
    pid = _canon_prop_id(prop_id)
    v = weights.get(pid, 0.0)
    if not v:
        v = weights.get(pid.replace("base", ""), 0.0)
    if not v:
        v = weights.get(pid + "base", 0.0)
    return v


def _top_weight_ids(weights: dict[str, float], count: int | None = None) -> frozenset[str]:
    """权重最高的词条 id 集合，用于高亮判定。

    count=None：只取「并列最高」的词条（主词条规则：如多条都=1 则全亮）。
    count=N：取权重最高的前 N 名（副词条规则：0.9/0.7/0.4/0.3/0.2/0.1 只亮 0.9/0.7/0.4/0.3），
              第 N 名若有并列一并算入。
    """
    positive = sorted((w for w in weights.values() if w > 0), reverse=True)
    if not positive:
        return frozenset()
    limit = positive[0] if count is None else positive[min(count, len(positive)) - 1]
    return frozenset(attr for attr, w in weights.items() if w >= limit and w > 0)


def _full_grade(total: float) -> str:
    for threshold, grade in FULL_GRADES:
        if total >= threshold:
            return grade
    return FULL_GRADES[-1][1]


def _piece_grade(score: float, max_score: float) -> str:
    if max_score <= 0:
        return PIECE_GRADES[-1][1]
    ratio = score / max_score
    for threshold, grade in PIECE_GRADES:
        if ratio >= threshold:
            return grade
    return PIECE_GRADES[-1][1]


class JiabailiScorer(BaseScorer):
    scorer_id = "jiabaili"
    meta = ScorerMeta(
        name="jiabaili",
        author="jiabaili",
        version="1.0.0",
        description="角色权重数据来源异环工坊",
    )

    def grades(self) -> Sequence[GradeSpec]:
        return (
            GradeSpec(id="ACE", color=(255, 208, 96), icon=_ASSETS / "rank_ACE.png"),
            GradeSpec(id="SSS", color=(255, 112, 112), icon=_ASSETS / "rank_SSS.png"),
            GradeSpec(id="SS", color=(255, 170, 96), icon=_ASSETS / "rank_SS.png"),
            GradeSpec(id="S", color=(255, 208, 96), icon=_ASSETS / "rank_S.png"),
            GradeSpec(id="A+", color=(200, 140, 250), icon=_ASSETS / "rank_A+.png"),
            GradeSpec(id="A", color=(170, 165, 240), icon=_ASSETS / "rank_A.png"),
            GradeSpec(id="B", color=(176, 182, 214), icon=_ASSETS / "rank_B.png"),
            GradeSpec(id="C", color=(120, 214, 162), icon=_ASSETS / "rank_C.png"),
            GradeSpec(id="D", color=(140, 150, 170), icon=_ASSETS / "rank_D.png"),
        )

    def describe_char(self, char_id: str) -> str:
        weights = _weights().get(str(char_id))
        if not weights:
            return ""
        sub = weights.get("sub") or {}
        main = weights.get("main") or {}
        names = _attr_names()
        lines = ["评分标准：350 满分制（核心件 150 + 驱动块面积 × 10），角色权重数据来源异环工坊"]
        if sub:
            top = sorted(sub.items(), key=lambda item: -item[1])[:5]
            lines.append(
                "有效副词条：" + "、".join(f"{names.get(attr, attr)}×{weight:g}" for attr, weight in top)
            )
        if main:
            lines.append("核心主词条：" + "、".join(names.get(attr, attr) for attr in sorted(main)))
        return "\n".join(lines)

    async def prepare(self) -> None:
        # 预热：强拉一次接口并落盘，再启动每天 4 整点（00/06/12/18）定时刷新
        await _refresh_weights(force=True)
        _start_daily_task()

    async def close(self) -> None:
        global _REMOTE_WEIGHTS
        _REMOTE_WEIGHTS = None
        _stop_daily_task()
        _cached_weights.cache_clear()
        _attr_names.cache_clear()
        _attr_name_ids.cache_clear()

    async def score_character(self, character: CharacterDetail) -> _Result | None:
        # 每次评分优先实时接口，失败自动降级到定时缓存文件
        await _refresh_weights()
        weights = _weights().get(str(character.id))
        if not weights:
            return None
        items = (*character.suit.core, *character.suit.pie) if character.suit.id else ()
        if not items:
            return None

        sub_weights: dict[str, float] = weights.get("sub") or {}  # type: ignore[assignment]
        main_weights: dict[str, float] = weights.get("main") or {}  # type: ignore[assignment]
        # 理论满权重 = 该角色副词条权重最高的 4 个之和
        max_weight = sum(sorted(sub_weights.values(), reverse=True)[:4])
        if max_weight <= 0:
            return None

        # 高亮口径：副词条 = 权重最高的 4 个（即理论满权重口径），主词条 = 权重最高（并列都亮）
        top_main_ids = _top_weight_ids(main_weights)
        top_sub_ids = _top_weight_ids(sub_weights, 4)

        total = 0.0
        equipment: list[_EquipmentView] = []
        for item in items:
            area = _piece_area(item)
            unlocked = item.lev // 5
            q = _quality_factor(item.id)
            if area is None:
                # 核心件（卡带）：(主词条权重 × 50 + 副词条权重 / 理论满权重 × 100) × 品质
                # 主词条数据缺失时按默认「魂属性异能伤害增强」取权重
                main_w = max(
                    (_weight_for(_resolve_main_prop_id(prop), main_weights) for prop in item.main_properties),
                    default=0.0,
                )
                sub_w = sum(
                    _weight_for(prop.id, sub_weights)
                    for prop in item.properties
                    if prop.value.strip()
                )
                piece_score = (main_w * CORE_MAIN_SCORE + sub_w / max_weight * CORE_SUB_SCORE) * q
                piece_max = CORE_MAX_SCORE
            else:
                # 盘件（驱动块）：(副词条权重 / 理论满权重 × 面积 × 10) × 品质
                sub_w = sum(
                    _weight_for(prop.id, sub_weights)
                    for prop in item.properties
                    if prop.value.strip()
                )
                piece_score = (sub_w / max_weight * area * PIE_SCORE_PER_AREA) * q
                piece_max = area * PIE_SCORE_PER_AREA

            piece_score = min(piece_score, piece_max)
            total += piece_score
            # 核心件与盘件均不挂评级徽章：grade 置 None，渲染端 grade_badge 对 None 不画
            piece_grade = None
            equipment.append(
                _EquipmentView(
                    item_id=item.id,
                    display=f"{piece_score:.1f}分",
                    grade=piece_grade,
                    unlocked_subs=unlocked,
                )
            )

        total = round(min(total, FULL_SET_SCORE), 1)
        return _Result(
            score=total,
            display=f"{total:g}",
            grade=_full_grade(total),
            equipment=tuple(equipment),
            main_weights=dict(main_weights),
            sub_weights=dict(sub_weights),
            top_main_ids=top_main_ids,
            top_sub_ids=top_sub_ids,
        )


register_scorer(JiabailiScorer())
