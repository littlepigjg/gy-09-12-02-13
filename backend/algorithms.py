"""图算法：BFS 最短路径、共同好友、简化 PageRank、简化 Louvain 社群发现。

所有算法均直接读取内存中的邻接表缓存（Graph.adj），
不访问数据库，保证在大规模图上的内存级高效执行。
"""

from collections import deque


def bfs_shortest_path(graph, source, target):
    """广度优先搜索求无权图最短路径。

    返回路径节点列表（含首尾），不可达返回 None。
    时间复杂度 O(V + E)。
    """
    source, target = str(source), str(target)
    if not graph.has_node(source) or not graph.has_node(target):
        return None
    if source == target:
        return [source]

    visited = {source}
    prev = {source: None}
    queue = deque([source])

    while queue:
        cur = queue.popleft()
        if cur == target:
            break
        for nb in graph.adj.get(cur, {}):
            if nb not in visited:
                visited.add(nb)
                prev[nb] = cur
                queue.append(nb)

    if target not in visited:
        return None

    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def common_friends(graph, a, b):
    """求两个节点的共同好友（共同邻居）。

    基于邻接表集合求交集，O(min(deg(a), deg(b)))。
    """
    a, b = str(a), str(b)
    if not graph.has_node(a) or not graph.has_node(b):
        return []
    return sorted(set(graph.adj.get(a, {})) & set(graph.adj.get(b, {})))


# 共同好友排序的评分权重：
#   关系强度（边权）45% + 朋友圈重叠（共同邻居 Jaccard）40% + 同社群 15%
W_TIE = 0.45
W_EMBED = 0.40
W_COMM = 0.15

RANKING_RULE = (
    "综合评分 = 45%×平均关系强度（与双方的边权按各自最高边权归一化）"
    " + 40%×平均朋友圈重叠度（共同邻居的 Jaccard 占比）"
    " + 15%×同社群加分；总分相同则按本人好友数（人脉广度）排序，仍相同按节点 ID 排序。"
)


def _normalized_strength(adj, x):
    """x 与各邻居的归一化关系强度（按 x 自身最高边权归一到 0~1）。

    返回 ({neighbor: strength}, max_weight)。边权全为 0（无强度信息）时
    存在连边即给中性分 0.5，避免该维度错误地把所有人压成 0。
    """
    nbs = adj.get(x, {})
    max_w = max(nbs.values(), default=0.0)
    if max_w <= 0:
        return {nb: 0.5 for nb in nbs}, 0.0
    return {nb: w / max_w for nb, w in nbs.items()}, max_w


def _jaccard(set_a, set_b):
    """两个邻居集合的 Jaccard 相似度（朋友圈重叠度），返回 (相似度, 交集大小)。"""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0, 0
    inter = set_a & set_b
    return len(inter) / union, len(inter)


def _summarize(tie, embed, comm, other_shared, degree):
    """根据各信号给出一句话的关系定性。"""
    if tie >= 0.75 and embed >= 0.12:
        return "与双方都是强关系，且朋友圈深度交织——最核心的共同密友"
    if tie >= 0.75:
        return "与双方的边权都很高，是你们之间最强的直接关系"
    if embed >= 0.2:
        return "边权不算突出，但朋友圈高度重叠，实际处在同一个紧密交往圈"
    if comm >= 1.0 and tie >= 0.4:
        return "与双方同属一个社群，关系中等偏上"
    if degree >= 10 and other_shared >= 3:
        return "人脉枢纽型：本人交友广，还与你们共享不少好友，是重要的关系桥梁"
    if tie < 0.35 and other_shared == 0:
        return "与双方边权都偏低、也没有其他共同好友，属于典型的点头之交"
    if tie < 0.35:
        return "与双方以弱联系为主，关系处于共同好友的外围"
    return "与双方都有一定直接联系，亲密程度居中"


def rank_common_friends(graph, a, b, community=None):
    """对共同好友按亲疏远近排序，并逐名给出可核对的量化解释。

    区分四种状态，绝不混淆：
    - node_missing：至少一个节点在图中不存在
    - same_node：两个参数是同一个节点
    - no_common：节点都存在、确为两人，但没有共同好友
    - ok：存在共同好友，ranked 为排序+解释后的完整名单
      （ranked 集合与旧版平铺名单 friends 完全一致，一个不少）。

    community 为可选的 Louvain 划分 {node: community_id}，由调用方缓存传入。
    """
    a, b = str(a), str(b)
    base = {
        "node1": a,
        "node2": b,
        "ranking_rule": RANKING_RULE,
        "friends": [],
        "count": 0,
        "ranked": [],
    }

    missing = [x for x in (a, b) if not graph.has_node(x)]
    if missing:
        existing = [x for x in (a, b) if graph.has_node(x)]
        if len(missing) == 2:
            detail = f"节点「{a}」和「{b}」在图中都不存在"
        else:
            detail = (
                f"节点「{missing[0]}」在图中不存在"
                + (f"（另一节点「{existing[0]}」存在）" if existing else "")
            )
        return {
            **base,
            "status": "node_missing",
            "missing_nodes": missing,
            "existing_nodes": existing,
            "message": (
                f"{detail}，无法比较共同好友。请注意这与「没有共同好友」是两回事："
                "前者是查无此人，后者是两个人都在图里、只是好友圈没有任何交集。"
            ),
        }

    if a == b:
        return {
            **base,
            "status": "same_node",
            "message": f"节点 A 与节点 B 都是「{a}」，是同一个人；共同好友比较需要两个不同的节点。",
        }

    adj = graph.adj
    common = set(adj.get(a, {})) & set(adj.get(b, {}))
    if not common:
        return {
            **base,
            "status": "no_common",
            "message": f"节点「{a}」与「{b}」都存在，但两人没有共同好友（邻接集合交集为空）。",
        }

    def name(nid):
        return graph.nodes.get(nid, {}).get("name") or nid

    na, nb = name(a), name(b)
    str_a, max_a = _normalized_strength(adj, a)
    str_b, max_b = _normalized_strength(adj, b)
    nbrs_a, nbrs_b = set(adj[a]), set(adj[b])
    ca = community.get(a) if community else None
    cb = community.get(b) if community else None

    rows = []
    for f in sorted(common):
        nbrs_f = set(adj[f])
        sa, sb = str_a[f], str_b[f]
        ja, ka = _jaccard(nbrs_a, nbrs_f)
        jb, kb = _jaccard(nbrs_b, nbrs_f)
        cf = community.get(f) if community else None
        same_a = cf is not None and cf == ca
        same_b = cf is not None and cf == cb

        tie = (sa + sb) / 2.0
        embed = (ja + jb) / 2.0
        comm_score = (float(same_a) + float(same_b)) / 2.0
        score = W_TIE * tie + W_EMBED * embed + W_COMM * comm_score

        # 同时认识 a、b、f 三方的人：f 作为「关系桥梁」的证据
        triple = len((nbrs_f & nbrs_a & nbrs_b) - {a, b})
        degree = len(nbrs_f)

        reasons = [
            f"与{na}：边权 {adj[a][f]:.2g}（{na} 的最高边权 {max_a:.2g}，相对强度 {round(sa * 100)}%）",
            f"与{nb}：边权 {adj[b][f]:.2g}（{nb} 的最高边权 {max_b:.2g}，相对强度 {round(sb * 100)}%）",
        ]
        if ka > 0:
            reasons.append(f"与{na}除彼此外还有 {ka} 个共同好友，邻居重叠度 {round(ja * 100)}%")
        else:
            reasons.append(f"与{na}除彼外侧没有其他共同好友，目前是单线联系")
        if kb > 0:
            reasons.append(f"与{nb}除彼外侧还有 {kb} 个共同好友，邻居重叠度 {round(jb * 100)}%")
        else:
            reasons.append(f"与{nb}除彼外侧没有其他共同好友，目前是单线联系")
        if community:
            if same_a:
                reasons.append(f"与{na}同属社群 {ca}，社群项加分")
            else:
                reasons.append(f"与{na}分属社群 {cf} / {ca}，社群项不加分")
            if same_b:
                reasons.append(f"与{nb}同属社群 {cb}，社群项加分")
            else:
                reasons.append(f"与{nb}分属社群 {cf} / {cb}，社群项不加分")
        bridge = f"，其中 {triple} 人同时也是{na}和{nb}的好友" if triple > 0 else ""
        reasons.append(f"本人共有 {degree} 个好友{bridge}（人脉广度，用于同分决胜）")

        rows.append(
            {
                "node": f,
                "name": name(f),
                "score_raw": score,
                "score": round(score, 4),
                "degree": degree,
                "triple_shared": triple,
                "tie": round(tie, 4),
                "embed": round(embed, 4),
                "comm": comm_score,
                "summary": _summarize(tie, embed, comm_score, ka + kb, degree),
                "reasons": reasons,
            }
        )

    # 总分降序 → 人脉广度降序 → ID 升序（确定性排序）
    rows.sort(key=lambda r: (-r["score_raw"], -r["degree"], r["node"]))

    # 位置解释需要回看上一名的原始分，统一在删字段之前生成
    for i, row in enumerate(rows):
        row["rank"] = i + 1
        if i == 0:
            note = (
                f"位列第 1：综合评分 {row['score']}，在全部 {len(rows)} 个共同好友中最高"
                "（评分依据见上方排序规则）。"
            )
        else:
            prev = rows[i - 1]
            gaps = {
                "关系强度": prev["tie"] - row["tie"],
                "朋友圈重叠度": prev["embed"] - row["embed"],
                "同社群加分": prev["comm"] - row["comm"],
            }
            main_gap = max(gaps, key=gaps.get)
            if abs(prev["score_raw"] - row["score_raw"]) < 1e-12:
                note = (
                    f"与上一名「{prev['name']}」总分相同，按人脉广度排序："
                    f"对方 {prev['degree']} 个好友，本人 {row['degree']} 个。"
                )
            elif gaps[main_gap] <= 1e-9:
                note = (
                    f"综合评分 {prev['score']} → {row['score']}，三项信号均小幅落后于"
                    f"上一名「{prev['name']}」。"
                )
            else:
                note = (
                    f"排在第 {i + 1}：评分 {row['score']}，低于上一名「{prev['name']}」"
                    f"（{prev['score']}），差距主要来自「{main_gap}」。"
                )
        if i == len(rows) - 1 and len(rows) > 1:
            note += " 评分垫底，是这批共同好友里关系最外围的一位。"
        row["position_note"] = note

    for row in rows:
        del row["score_raw"]  # 内部排序用原始分，不输出

    return {
        **base,
        "node1_name": na,
        "node2_name": nb,
        "status": "ok",
        "message": f"{na} 与 {nb} 共有 {len(rows)} 个共同好友，已按亲疏远近排序。",
        "friends": [r["node"] for r in rows],
        "count": len(rows),
        "ranked": rows,
    }


def pagerank(graph, damping=0.85, epsilon=1e-6, max_iter=100):
    """简化 PageRank（幂迭代法）。

    将无向图视为双向图，按出度均分权重。
    时间复杂度 O(iter * E)。
    """
    nodes = graph.node_ids()
    n = len(nodes)
    if n == 0:
        return {}

    ranks = {node: 1.0 / n for node in nodes}

    for _ in range(max_iter):
        base = (1.0 - damping) / n
        new_ranks = {node: base for node in nodes}
        for node in nodes:
            nbs = graph.adj.get(node, {})
            if not nbs:
                # 悬挂节点：权重均匀回流到所有节点
                share = ranks[node] / n
                for other in nodes:
                    new_ranks[other] += damping * share
            else:
                share = ranks[node] / len(nbs)
                for nb in nbs:
                    new_ranks[nb] += damping * share

        delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        if delta < epsilon:
            break

    return ranks


def _local_moving(cur_nodes, weight, deg, m, comm, tot, max_passes):
    """单层局部移动：将每个节点移动到使模块度增益最大的邻居社区。

    返回本次移动是否改进了划分。
    """
    moved_any = False
    for _ in range(max_passes):
        moved = False
        for u in cur_nodes:
            cu = comm[u]

            # 统计 u 与各邻居社区的连接权重（不含自环）
            w_to = {}
            for v, w in weight[u].items():
                c = comm[v]
                w_to[c] = w_to.get(c, 0.0) + w

            # 从当前社区移除 u
            tot[cu] -= deg[u]

            best_c = cu
            best_gain = 0.0
            for c, k_uin in w_to.items():
                gain = k_uin / m - tot[c] * deg[u] / (2.0 * m * m)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c

            # 将 u 加入最佳社区
            comm[u] = best_c
            tot[best_c] += deg[u]

            if best_c != cu:
                moved = True

        if not moved:
            break
        moved_any = True
    return moved_any


def louvain(graph, max_levels=20, max_passes=20):
    """Louvain 社群发现（局部移动 + 社区聚合两阶段迭代）。

    在加权无向图上多层级执行：
    1. 局部移动：每个节点初始独立成社区，按模块度增益
       ΔQ = k_i,in / m - Σ_tot * k_i / (2m^2) 反复移动节点。
    2. 社区聚合：将每个社区收缩为一个超节点，形成更粗粒度的图，
       再递归执行局部移动。

    相比完整实现省略了部分优化（如随机化节点顺序、模块度阈值早停），
    但保留了两阶段核心结构，可稳定得到合理的社群划分。
    """
    nodes = graph.node_ids()
    if not nodes:
        return {}

    adj = graph.adj
    m2 = sum(sum(w for w in adj[u].values()) for u in nodes)
    if m2 == 0:
        return {u: 0 for u in nodes}
    m = m2 / 2.0

    # 当前层级的超节点（初始为原始节点）
    cur_nodes = list(nodes)
    # 原始节点 -> 当前超节点
    node_to_super = {u: u for u in nodes}

    # 当前层级的加权图表示
    weight = {u: dict(adj[u]) for u in cur_nodes}          # 节点间边权（无自环）
    self_w = {u: 0.0 for u in cur_nodes}                   # 自环权（社区内部边权）
    deg = {u: sum(adj[u].values()) for u in cur_nodes}     # 总度数

    for _ in range(max_levels):
        # ---- 局部移动 ----
        comm = {u: i for i, u in enumerate(cur_nodes)}
        tot = {i: deg[u] for i, u in enumerate(cur_nodes)}
        improved = _local_moving(cur_nodes, weight, deg, m, comm, tot, max_passes)

        # 若未能进一步合并，则停止
        if not improved or len(set(comm.values())) == len(cur_nodes):
            break

        # ---- 社区聚合 ----
        mapping = {}
        for u in cur_nodes:
            c = comm[u]
            if c not in mapping:
                mapping[c] = len(mapping)
        new_nodes = list(range(len(mapping)))

        new_weight = {c: {} for c in new_nodes}
        new_self = {c: 0.0 for c in new_nodes}
        new_deg = {c: 0.0 for c in new_nodes}

        # 聚合自环与总度数
        for u in cur_nodes:
            c = mapping[comm[u]]
            new_self[c] += self_w[u]
            new_deg[c] += deg[u]

        # 聚合节点间边（每条无向边只统计一次）
        seen = set()
        for u in cur_nodes:
            cu = mapping[comm[u]]
            for v, w in weight[u].items():
                pair = (u, v) if u <= v else (v, u)
                if pair in seen:
                    continue
                seen.add(pair)
                cv = mapping[comm[v]]
                if cu == cv:
                    new_self[cu] += w
                else:
                    new_weight[cu][cv] = new_weight[cu].get(cv, 0.0) + w
                    new_weight[cv][cu] = new_weight[cv].get(cu, 0.0) + w

        # 更新「原始节点 -> 超节点」映射，进入下一层级
        for o in nodes:
            node_to_super[o] = mapping[comm[node_to_super[o]]]
        cur_nodes = new_nodes
        weight = new_weight
        self_w = new_self
        deg = new_deg

    # 将最终划分映射回原始节点并压缩编号
    final_comm = {u: comm[node_to_super[u]] for u in nodes}
    mapping = {}
    result = {}
    for u in nodes:
        c = final_comm[u]
        if c not in mapping:
            mapping[c] = len(mapping)
        result[u] = mapping[c]
    return result


def community_groups(community):
    """将 {node: community_id} 转换为 {community_id: [nodes]}。"""
    groups = {}
    for node, cid in community.items():
        groups.setdefault(cid, []).append(node)
    return groups
