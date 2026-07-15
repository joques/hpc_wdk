using MLJ
import MLJLinearModels
using BSON: @save, @load
using GLM
import RDatasets: dataset
import DataFrames: DataFrame, describe, nrow, select, Not, rename!

#load dataset
boston = dataset("MASS", "Boston");

# Prepare the dataset

data = coerce(boston, autotype(boston, :discrete_to_continuous));

# Check variable significance with p-values
#
# GLM provides the usual regression coefficient table: estimate, standard error,
# t-statistic, and p-value. Variables with p-values below 0.05 are used for
# training the MLJ model below.
significance_model = lm(
	@formula(MedV ~ Crim + Zn + Indus + Chas + NOx + Rm + Age + Dis + Rad + Tax + PTRatio + Black + LStat),
	boston,
)

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
significance_rows = [
	(
		Variable = significance_table.Variable[i],
		Coefficient = significance_table.Coefficient[i],
		StdError = significance_table.StdError[i],
		TStatistic = significance_table.TStatistic[i],
		PValue = significance_table.PValue[i],
	)
	for i in 1:nrow(significance_table)
]
significant_rows = [
	(
		Variable = significant_variables.Variable[i],
		Coefficient = significant_variables.Coefficient[i],
		StdError = significant_variables.StdError[i],
		TStatistic = significant_variables.TStatistic[i],
		PValue = significant_variables.PValue[i],
	)
	for i in 1:nrow(significant_variables)
]

println("Variable significance table:")
println(significance_table)
println()
println("Variables significant at the 5% level:")
println(significant_variables)

@save "significance.bson" significance_rows significant_rows significant_variable_names

begin
	y = data.MedV;
	X = select(data, significant_variable_names);
end

# Train the model

LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
mdl = LinearRegressor()
begin
	mach = machine(mdl, X, y)
	fit!(mach)
end

@save "model.bson" mach

#Evaluate the Model using Root Mean Squared Error
ŷ = MLJ.predict(mach, X)


@save "result.bson" round(rms(ŷ, y), sigdigits=4)
