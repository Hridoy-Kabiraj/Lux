using Lux,
    Boltz,
    ComponentArrays,
    OrdinaryDiffEqVerner,
    Optimization,
    OptimizationOptimJL,
    OptimizationOptimisers,
    SciMLSensitivity,
    Statistics,
    Printf,
    Random
using DynamicExpressions, SymbolicRegression, MLJ, SymbolicUtils, Latexify
using CairoMakie

function plot_dynamics(sol, us, ts)
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel=L"t")
    ylims!(ax, (-6, 6))

    lines!(ax, ts, sol[1, :]; label=L"u_1(t)", linewidth=3)
    lines!(ax, ts, sol[2, :]; label=L"u_2(t)", linewidth=3)

    lines!(ax, ts, vec(us); label=L"u(t)", linewidth=3)

    axislegend(ax; position=:rb)

    return fig
end

# Training a Neural Network based UDE
rng = Xoshiro(0)
tspan = (0.0, 8.0)

mlp = Chain(Dense(1 => 4, gelu), Dense(4 => 4, gelu), Dense(4 => 1))

function construct_ude(mlp, solver; kwargs...)
    return @compact(; mlp, solver, kwargs...) do x_in, ps
        x, ts, ret_sol = x_in

        function dudt(du, u, p, t)
            u₁, u₂ = u
            du[1] = u₂
            du[2] = mlp([t], p)[1]^3
            return nothing
        end

        prob = ODEProblem{true}(dudt, x, extrema(ts), ps.mlp)

        sol = solve(
            prob,
            solver;
            saveat=ts,
            sensealg=QuadratureAdjoint(; autojacvec=ReverseDiffVJP(true)),
            kwargs...,
        )

        us = mlp(reshape(ts, 1, :), ps.mlp)
        ret_sol === Val(true) && @return sol, us
        @return Array(sol), us
    end
end

ude = construct_ude(mlp, Vern9(); abstol=1e-10, reltol=1e-10);



ude_test = construct_ude(mlp, Vern9(); abstol=1e-10, reltol=1e-10);

function train_model_1(ude, rng, ts_)
    ps, st = Lux.setup(rng, ude)
    ps = ComponentArray{Float64}(ps)
    stateful_ude = StatefulLuxLayer{true}(ude, nothing, st)

    ts = collect(ts_)

    function loss_adjoint(θ)
        x, us = stateful_ude(([-4.0, 0.0], ts, Val(false)), θ)
        return mean(abs2, 4 .- x[1, :]) + 2 * mean(abs2, x[2, :]) + 0.1 * mean(abs2, us)
    end

    callback = function (state, l)
        state.iter % 50 == 1 && @printf "Iteration: %5d\tLoss: %10g\n" state.iter l
        return false
    end

    optf = OptimizationFunction((x, p) -> loss_adjoint(x), AutoZygote())
    optprob = OptimizationProblem(optf, ps)
    res1 = solve(optprob, Optimisers.Adam(0.001); callback, maxiters=500)

    optprob = OptimizationProblem(optf, res1.u)
    res2 = solve(optprob, LBFGS(); callback, maxiters=100)

    return StatefulLuxLayer{true}(ude, res2.u, st)
end

trained_ude = train_model_1(ude, rng, 0.0:0.01:8.0)

sol, us = ude_test(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), trained_ude.ps, trained_ude.st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)


function train_model_2(stateful_ude::StatefulLuxLayer, ts_)
    ts = collect(ts_)

    function loss_adjoint(θ)
        x, us = stateful_ude(([-4.0, 0.0], ts, Val(false)), θ)
        return mean(abs2, 4 .- x[1, :]) .+ 2 * mean(abs2, x[2, :]) .+ mean(abs2, us)
    end

    callback = function (state, l)
        state.iter % 10 == 1 && @printf "Iteration: %5d\tLoss: %10g\n" state.iter l
        return false
    end

    optf = OptimizationFunction((x, p) -> loss_adjoint(x), AutoZygote())
    optprob = OptimizationProblem(optf, stateful_ude.ps)
    res2 = solve(optprob, LBFGS(); callback, maxiters=100)

    return StatefulLuxLayer{true}(stateful_ude.model, res2.u, stateful_ude.st)
end

trained_ude = train_model_2(trained_ude, 0.0:0.01:8.0)

sol, us = ude_test(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), trained_ude.ps, trained_ude.st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)


# Symbolic Regression

# Data Generation for Symbolic Regression
ts = reshape(collect(0.0:0.1:8.0), 1, :)

X_train = mlp[1](ts, trained_ude.ps.mlp.layer_1, trained_ude.st.mlp.layer_1)[1]

Y_train = mlp[2](X_train, trained_ude.ps.mlp.layer_2, trained_ude.st.mlp.layer_2)[1]

# Fitting the Symbolic Expression 
srmodel = MultitargetSRRegressor(;
    binary_operators=[+, -, *, /], niterations=100, save_to_file=false
);


mach = machine(srmodel, X_train', Y_train')
fit!(mach; verbosity=0)
r = report(mach)
best_eq = [
    r.equations[1][r.best_idx[1]],
    r.equations[2][r.best_idx[2]],
    r.equations[3][r.best_idx[3]],
    r.equations[4][r.best_idx[4]],
]

hybrid_mlp = Chain(
    Dense(1 => 4, gelu),
    Layers.DynamicExpressionsLayer(OperatorEnum(; binary_operators=[+, -, *, /]), best_eq),
    Dense(4 => 1),
)

hybrid_ude = construct_ude(hybrid_mlp, Vern9(); abstol=1e-10, reltol=1e-10);

st = Lux.initialstates(rng, hybrid_ude)
ps = (;
    mlp=(;
        layer_1=trained_ude.ps.mlp.layer_1,
        layer_2=Lux.initialparameters(rng, hybrid_mlp[2]),
        layer_3=trained_ude.ps.mlp.layer_3,
    )
)
ps = ComponentArray(ps)

sol, us = hybrid_ude(([-4.0, 0.0], 0.0:0.01:8.0, Val(true)), ps, st)[1];
plot_dynamics(sol, us, 0.0:0.01:8.0)