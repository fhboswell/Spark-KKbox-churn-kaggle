# Databricks notebook source
sqlDF = spark.sql("SELECT * FROM transactions")
display(sqlDF)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
    inputCols=["payment_method_id", "plan_list_price", "actual_amount_paid"], outputCol="features")
final_df = assembler.transform(sqlDF)
final_df.select("features").show(4, False)



# COMMAND ----------

from pyspark.ml.classification import LogisticRegression


lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol="is_cancel")

# Fit the model
lrModel = lr.fit(final_df)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial", labelCol="is_cancel")

# Fit the model
mlrModel = mlr.fit(final_df)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))

# COMMAND ----------

from pyspark.ml.feature import RFormula

rf = RFormula(formula="~ payment_method_id +  payment_plan_days + plan_list_price - 1")
final_df_rf = rf.fit(sqlDF).transform(sqlDF)
final_df_rf.select("features").show(4, False)

# COMMAND ----------


