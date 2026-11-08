# Copyright (c) 2026 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Author: Leonard Schmidt

from typing import Optional

from core.types import Container, EdgeKind, OperatorKind, Role, TemplateEdge, TemplateVertex
from ir.rpeg import RPEG
from model.models import Model, ModelSuite
from model.workload import Workload


def is_AC_share_applicable(rpeg: RPEG) -> bool:
    """
    Precondition for Rule S3: return True iff both ACTOR and CRITIC vertices are present in the template.

    :param rpeg: RLM Pipeline Execution Graph.
    :type rpeg: RPEG
    :return: True, when both actor and critic are part of the steady-state graph IR.
    :rtype: bool
    """
    return Model.ACTOR in rpeg.trainable and Model.CRITIC in rpeg.trainable


def build_AC_share_RPEG(suite: ModelSuite, workload: Workload, tau: Optional[int] = None, verbose: bool = False) -> RPEG:
    """
    Construct the AC-shared rewrite of the PPO template.

    :param suite: Model hyperparameters.
    :type suite: ModelSuite
    :param workload: Workload settings.
    :type workload: Workload
    :param tau: Staleness budget. Defaults to None.
    :type tau: Optional[int]
    :param verbose: Verbose flag for additional output. Defaults to False.
    :type verbose: bool
    :return: RLM Pipeline Execution Graph.
    :rtype: RPEG
    """
    B, K, S, T = workload.B, workload.K, workload.S, workload.T  
    trainable = {Model.AC_SHARE}

    # Transient containers (within-step data)
    X = Container(name="prompts", count=B * S, elem_size_bytes=2)
    Y = Container(name="completions", count=B * K * (S + T), elem_size_bytes=2)
    log_ac = Container(name="log_ac", count=B * K * T, elem_size_bytes=2)
    log_ref = Container(name="log_ref", count=B * K * T, elem_size_bytes=2)
    r = Container(name="rewards", count=B * K, elem_size_bytes=2)
    v = Container(name="values", count=B * K * T, elem_size_bytes=2)

    # Weight containers (cross-step via publication edges)
    W_AC = Container(name="W_AC", count=suite.n_params(Model.AC_SHARE), elem_size_bytes=2)
    W_ref = Container(name="W_ref", count=suite.n_params(Model.REFERENCE), elem_size_bytes=2)
    W_R = Container(name="W_R", count=suite.n_params(Model.REWARD), elem_size_bytes=2)

    containers = (Y, log_ac, log_ref, r, v, W_AC, W_ref, W_R)
    in_containers = (X, W_ref, W_R)
    out_containers = () 

    # Operator kinds
    o1 = OperatorKind(role=Role.GENERATION, model=Model.AC_SHARE, read=(W_AC, X), write=(Y, log_ac))
    o2 = OperatorKind(role=Role.INFERENCE, model=Model.REFERENCE, read=(W_ref, Y), write=(log_ref, ))
    o3 = OperatorKind(role=Role.INFERENCE, model=Model.REWARD, read=(W_R, Y), write=(r, ))
    o4 = OperatorKind(role=Role.INFERENCE, model=Model.AC_SHARE, read=(W_AC, Y), write=(v, ))
    o5 = OperatorKind(role=Role.TRAINING, model=Model.AC_SHARE, read=(W_AC, Y, log_ac, log_ref, r, v), write=(W_AC, ))

    # Vertices
    g_ac = TemplateVertex(id="g_ac", kind=o1)
    i_ref = TemplateVertex(id="i_ref", kind=o2)
    i_R = TemplateVertex(id="i_R", kind=o3)
    i_ac = TemplateVertex(id="i_ac", kind=o4)
    t_ac = TemplateVertex(id="t_ac", kind=o5)

    vertices = (g_ac, i_ref, i_R, i_ac, t_ac) 

    # Edges
    # generation -> inference
    e1 = TemplateEdge(src=g_ac, dst=i_ref, container=Y, kind=EdgeKind.DATA)
    e2 = TemplateEdge(src=g_ac, dst=i_R, container=Y, kind=EdgeKind.DATA)
    e3 = TemplateEdge(src=g_ac, dst=i_ac, container=Y, kind=EdgeKind.DATA)

     # inference -> training
    e4 = TemplateEdge(src=i_ref, dst=t_ac, container=log_ref, kind=EdgeKind.DATA)
    e5 = TemplateEdge(src=i_R, dst=t_ac, container=r, kind=EdgeKind.DATA)
    e6 = TemplateEdge(src=i_ac, dst=t_ac, container=v, kind=EdgeKind.DATA)

    # training -> generation | inference
    e7 = TemplateEdge(src=t_ac, dst=g_ac, container=W_AC, kind=EdgeKind.PUB)
    e8 = TemplateEdge(src=t_ac, dst=i_ac, container=W_AC, kind=EdgeKind.PUB)

    data_edges = (e1, e2, e3, e4, e5, e6)
    pub_edges = (e7, e8) 

    # Staleness
    # Default: synchronous (τ=0). With τ>0 the actor inference could run on stale weights while the
    # previous training step finishes.
    if tau is None: 
        tau = {Model.AC_SHARE: 0} 

    # AC-Share PPO RPEG
    ac_share_rpeg = RPEG(vertices, data_edges, pub_edges, containers, in_containers, out_containers, tau, trainable)
    
    ac_share_rpeg.well_formed(verbose)

    if verbose:
        print("RPEG instantiated successfully for AC-Share PPO.")
        print(f"    Template Nodes: {len(ac_share_rpeg.vertices)}.")
        print(f"    Data edges: {len(ac_share_rpeg.data_edges)}.")
        print(f"    Publication edges: {len(ac_share_rpeg.pub_edges)}.")
        print(f"    Container: {len(ac_share_rpeg.containers)}.")
        print(f"    IN_Container: {len(ac_share_rpeg.in_containers)}.")
        print(f"    OUT_Container: {len(ac_share_rpeg.out_containers)}.")

    return ac_share_rpeg
