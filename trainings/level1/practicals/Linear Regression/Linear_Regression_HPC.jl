#!/usr/bin/env julia

using Distributed

# Start distributed workers for either a Slurm allocation or a local machine.
# The local fallback keeps the worker count modest so the example remains usable
# on laptops while still exercising Julia's distributed execution path.
function setup_workers()
    nworkers() > 1 && return

    if haskey(ENV, "SLURM_JOB_ID")
        try
            # SlurmClusterManager starts workers with srun so workers can run
            # on every node allocated to the job.
            @eval import SlurmClusterManager
            manager = Base.invokelatest(SlurmClusterManager.SlurmManager)
            Base.invokelatest(
                addprocs,
                manager;
                exeflags="--project=$(Base.active_project())",
            )
	catch err
            message = sprint(showerror, err)
            if err isa ArgumentError && occursin("Package SlurmClusterManager", message)
                error("""
                SlurmClusterManager is required for multi-node Slurm jobs.
                Install it once in this environment with:
                    julia --project=. -e 'using Pkg; Pkg.add("SlurmClusterManager")'
                Then resubmit:
                    sbatch Linear/linear_regression.slurm
                """)
            end
            rethrow()
        end
    else
        target_procs = max(2, min(Sys.CPU_THREADS, 6))
        addprocs(target_procs - nprocs(); exeflags="--project=$(Base.active_project())")
    end
end

setup_workers()

# Load the modeling and data packages on every process, since pmap may run any
# of the helper functions below on a worker rather than on the main process.
@everywhere begin
    using Sockets: gethostname
    using MLJ
    import MLJLinearModels
    using GLM
    import DataFrames: DataFrame, nrow, select
    import RDatasets: dataset
end


using BSON: @save
using Random
import DataFrames: DataFrame, nrow, select

@everywhere function significant_predictors(boston)
    # Fit the full GLM once to identify predictors whose p-values pass the
    # 5% threshold. The distributed training stage uses only these columns.
    formula = @formula(
        MedV ~ Crim + Zn + Indus + Chas + NOx + Rm + Age + Dis + Rad + Tax + PTRatio + Black + LStat
    )
    significance_model = lm(formula, boston)
    coef_table = coeftable(significance_model)

    significance_table = DataFrame(
        Variable = coef_table.rownms,
        Coefficient = coef_table.cols[1],
        StdError = coef_table.cols[2],
        TStatistic = coef_table.cols[3],
        PValue = coef_table.cols[4],
    )

    significant_variables = significance_table[
        (significance_table.Variable .!= "(Intercept)") .& (significance_table.PValue .< 0.05),
        :,
    ]

    significant_variable_names = Symbol.(significant_variables.Variable)
    return significance_table, significant_variables, significant_variable_names
end

@everywhere function row_records(table)
    # Convert DataFrame rows to plain named tuples before writing BSON files.
    # This keeps the saved result simple to inspect or reload in another script.
    return [
        (
            Variable = table.Variable[i],
            Coefficient = table.Coefficient[i],
            StdError = table.StdError[i],
            TStatistic = table.TStatistic[i],
            PValue = table.PValue[i],
        )
        for i in 1:nrow(table)
    ]
end


@everywhere function fit_and_score(train_rows, test_rows, predictor_names)
    # Each fold is independent, so workers can load the Boston data, train on
    # their assigned rows, and return only the RMSE for that split.
    boston = dataset("MASS", "Boston")
    data = coerce(boston, autotype(boston, :discrete_to_continuous))

    y = data.MedV
    X = select(data, predictor_names)

    LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
    mach = machine(LinearRegressor(), X[train_rows, :], y[train_rows])
    fit!(mach; verbosity=0)

    predictions = MLJ.predict(mach, X[test_rows, :])
    return rms(predictions, y[test_rows])
end

@everywhere function final_sufficient_stats(rows, predictor_names)
    # Compute the normal-equation pieces for one row chunk. Summing X'X and X'y
    # across chunks is enough to recover the same OLS solution as one full pass.
    boston = dataset("MASS", "Boston")
    data = coerce(boston, autotype(boston, :discrete_to_continuous))

    X = Float64.(Matrix(select(data, predictor_names)[rows, :]))
    y = Float64.(data.MedV[rows])
    design = hcat(ones(length(rows)), X)

    return (
        xtx = design' * design,
        xty = design' * y,
        n = length(rows),
    )
end

@everywhere function prediction_sse(rows, predictor_names, coefficients)
    # Score a chunk against the final distributed coefficients; the driver
    # combines these partial SSE values into one full-data RMSE.
    boston = dataset("MASS", "Boston")
    data = coerce(boston, autotype(boston, :discrete_to_continuous))

    X = Float64.(Matrix(select(data, predictor_names)[rows, :]))
    y = Float64.(data.MedV[rows])
    design = hcat(ones(length(rows)), X)
    residuals = design * coefficients - y

    return (
        sse = sum(abs2, residuals),
        n = length(rows),
    )
end

function kfolds(n; k=5, rng=MersenneTwister(42))
    # Shuffle deterministically so repeated HPC and local runs compare cleanly.
    shuffled = shuffle(rng, collect(1:n))
    fold_size = ceil(Int, n / k)

    return [
        begin
            test_rows = shuffled[start:min(start + fold_size - 1, n)]
            train_rows = setdiff(shuffled, test_rows)
            (train_rows=train_rows, test_rows=test_rows)
        end
	for start in 1:fold_size:n
    ]
end

function row_chunks(n, chunk_count)
    # Split row indices into approximately even chunks for worker-level pmap
    # tasks. The chunk count is capped by the number of available rows.
    chunk_count = max(1, min(chunk_count, n))
    chunk_size = ceil(Int, n / chunk_count)

    return [
	collect(start:min(start + chunk_size - 1, n))
        for start in 1:chunk_size:n
    ]
end

println("Distributed linear regression")
println("Processes: $(nprocs()) total, $(nworkers()) workers")
println("Process locations:")
# Capture the host for every process so Slurm runs can confirm work actually
# reached the allocated nodes.
process_locations = pmap(pid -> (pid=pid, host=fetch(@spawnat pid gethostname())), procs())
for location in process_locations
    println("  process $(location.pid) on $(location.host)")
end

boston = dataset("MASS", "Boston")
data = coerce(boston, autotype(boston, :discrete_to_continuous))

significance_table, significant_variables, significant_variable_names = significant_predictors(boston)
significance_rows = row_records(significance_table)
significant_rows = row_records(significant_variables)

println()
println("Variable significance table:")
println(significance_table)
println()
println("Variables significant at the 5% level:")
println(significant_variables)

# Save the significance analysis separately from the model artifacts so it can
# be reviewed without loading the distributed model output.
@save "Linear/significance_hpc.bson" significance_rows significant_rows significant_variable_names

# Cross-validation is embarrassingly parallel: each worker receives a fold and
# returns one validation RMSE.
folds = kfolds(nrow(data); k=5)
fold_scores = pmap(
    fold -> fit_and_score(fold.train_rows, fold.test_rows, significant_variable_names),
    folds,
)

mean_rmse = mean(fold_scores)
println()
println("5-fold RMSE values: ", round.(fold_scores, sigdigits=4))
println("Mean 5-fold RMSE: ", round(mean_rmse, sigdigits=4))

# Build the final model from distributed sufficient statistics instead of
# shipping the whole design matrix back from every worker.
training_chunks = row_chunks(nrow(data), nworkers())
partial_stats = pmap(
    rows -> final_sufficient_stats(rows, significant_variable_names),
    training_chunks,
)

xtx = reduce(+, map(part -> part.xtx, partial_stats))
xty = reduce(+, map(part -> part.xty, partial_stats))
coefficients = xtx \ xty

# Evaluate the final coefficients over the same row chunks to avoid collecting
# all predictions on the driver process.
partial_errors = pmap(
    rows -> prediction_sse(rows, significant_variable_names, coefficients),
    training_chunks,
)
full_sse = sum(part -> part.sse, partial_errors)
full_n = sum(part -> part.n, partial_errors)
full_rmse = sqrt(full_sse / full_n)


feature_names = ["(Intercept)"; string.(significant_variable_names)]
# Store enough metadata with the coefficient vector to understand where the
# distributed run executed and which predictors were kept.
distributed_model = (
    model_type = "DistributedOLSLinearRegression",
    feature_names = feature_names,
    intercept = coefficients[1],
    coefficients = coefficients,
    predictor_names = significant_variable_names,
    process_locations = process_locations,
    training_chunks = length(training_chunks),
)

@save "Linear/model_hpc.bson" distributed_model
@save "Linear/result_hpc.bson" fold_scores mean_rmse full_rmse

println()
println("Distributed final model coefficients:")
for (name, coefficient) in zip(feature_names, coefficients)
    println("  $(name): $(round(coefficient, sigdigits=6))")
end
println("Full-data RMSE: ", round(full_rmse, sigdigits=4))
