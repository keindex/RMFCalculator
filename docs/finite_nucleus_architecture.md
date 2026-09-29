# 有限核计算程序架构

本文对应 `RMFCalculator.eos.finite_nucleus` 当前实现。

```mermaid
flowchart TD
    API["调用入口<br/>compute_finite_nucleus(Z, N, params)"]
    WRAPPERS["便捷入口<br/>compute_Pb208 / compute_Sn132"]
    PARAMS["参数与常数<br/>NucleusParams / PARAMSETS<br/>FSUGold / FSU_Delta67"]
    UNIT["单位换算<br/>eos.unit.fm_MeV"]
    SOLVER["主控模块 solver<br/>创建径向网格、初始化状态、控制迭代"]

    API --> SOLVER
    WRAPPERS --> PARAMS
    WRAPPERS --> API
    PARAMS --> SOLVER
    UNIT --> SOLVER

    subgraph INIT["初始化"]
        DENSITY["density<br/>核半径与 Woods-Saxon 初始密度"]
        INITFIELDS["初始场<br/>σ、ω、ρ₀³、δ、A₀"]
        DENSITY --> INITFIELDS
    end
    SOLVER --> DENSITY
    INITFIELDS --> SOURCE

    subgraph LOOP["RMF 自洽迭代"]
        SOURCE["由当前密度和场构造场方程源项"]
        GF["green_function<br/>求解 σ、ω、ρ、δ 介子场与库仑场"]
        MIX["阻尼混合新旧场<br/>α = 0.5"]
        DIRAC["dirac_solver<br/>构造径向 Dirac Hamiltonian<br/>求本征态并填充质子、中子轨道"]
        NEWDENS["由占据轨道重建密度<br/>nₛ、nᵥ、n₃、n₃ₛ、nγ"]
        ENERGY["binding_energy<br/>计算总能量"]
        CHECK{"能量变化小于收敛阈值？"}

        SOURCE --> GF --> MIX --> DIRAC --> NEWDENS
        NEWDENS --> SOURCE
        MIX --> ENERGY --> CHECK
        CHECK -- "否，继续迭代" --> SOURCE
    end

    CHECK -- "是" --> RESULT
    RESULT["返回结果<br/>场分布、密度、轨道、总能量、均方根半径、收敛状态。。。。"]
```

## 模块职责

- `solver`：创建径向网格，协调场与密度更新，并检查能量收敛。
- `density`：提供核半径和 Woods-Saxon 初始密度；迭代中的密度由 Dirac 轨道重建。
- `green_function`：根据源项求解介子场和库仑场。
- `dirac_solver`：在给定平均场中求解单粒子态，填充质子和中子轨道并生成密度。
- `binding_energy`：计算总能量，用于自洽收敛判断及最终结果。
- `params`：定义核物质模型参数集。

## 能量计算说明

当前总能量由液滴模型项、场能项和质心修正组成。Dirac 轨道能量用于构造轨道与更新密度，没有直接作为单粒子能量项加入总能量。