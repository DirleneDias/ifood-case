# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType

# COMMAND ----------

# Criar sessão Spark
spark = SparkSession.builder.appName("EDA_Analysis").getOrCreate()

# COMMAND ----------

# Carregar
df = spark.read.format("delta").load("/data/processed/df_processado.delta")
df.printSchema()

# COMMAND ----------

## 1. Pré-processamento dos dados

# Deletar as colunas irrelevantes (excluindo colunas do tipo string)
df = df.drop("account_id", "event", "offer_id", "offer_type", "time_since_test_start", "email", "total_channels")

# Contar a distribuição das classes
total_count = df.count()
class_counts = df.groupBy("target").count().collect()
count_0 = class_counts[0]["count"] if class_counts[0]["target"] == 0 else class_counts[1]["count"]
count_1 = class_counts[1]["count"] if class_counts[1]["target"] == 1 else class_counts[0]["count"]

# Criar coluna de pesos inversamente proporcionais
minority_weight = count_0 / total_count  # Classe 1 recebe mais peso
majority_weight = count_1 / total_count  # Classe 0 recebe menos peso

df = df.withColumn("weight", when(col("target") == 1, minority_weight).otherwise(majority_weight))

# COMMAND ----------

## 2. Normalizar variáveis numéricas e Criar um vetor das variáveis
numeric_cols = ["discount_value", "duration", "min_value"]
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
df_vector = assembler.transform(df)
scaler = MinMaxScaler(inputCol="numeric_features", outputCol="features")
scaler_model = scaler.fit(df_vector)
df = scaler_model.transform(df_vector)

# COMMAND ----------

## 3. Dividir em treino e teste
train, test = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

## 4. Criar e treinar modelos com peso

# Modelo 1: Regressão Logística (com weightCol)
lr = LogisticRegression(featuresCol="features", labelCol="target", weightCol="weight", maxIter=10)

# Modelo 2: Random Forest (com weightCol)
rf = RandomForestClassifier(featuresCol="features", labelCol="target", weightCol="weight", numTrees=20)

# Criar pipelines
lr_pipeline = Pipeline(stages=[lr])
rf_pipeline = Pipeline(stages=[rf])

# Treinar modelos
lr_model = lr_pipeline.fit(train)
rf_model = rf_pipeline.fit(train)

# COMMAND ----------

## 5. Fazer previsões
lr_predictions = lr_model.transform(test)
rf_predictions = rf_model.transform(test)


# COMMAND ----------

## 6. Avaliar modelos

# Métricas de avaliação
evaluator_auc = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
evaluator_pr = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderPR")

# Calcular métricas para Regressão Logística
lr_auc = evaluator_auc.evaluate(lr_predictions)
lr_pr = evaluator_pr.evaluate(lr_predictions)

# Calcular métricas para Random Forest
rf_auc = evaluator_auc.evaluate(rf_predictions)
rf_pr = evaluator_pr.evaluate(rf_predictions)

# Mostrar resultados
print(f"""
Resultados dos Modelos:
--------------------------------------------------
Regressão Logística:
- AUC: {lr_auc:.4f}
- AUC-PR: {lr_pr:.4f}

Random Forest:
- AUC: {rf_auc:.4f}
- AUC-PR: {rf_pr:.4f}
--------------------------------------------------
""")

# COMMAND ----------

## 7. Otimização de hiperparâmetros (opcional)

# Exemplo para Random Forest
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [10, 20, 30])
             .addGrid(rf.maxDepth, [5, 10, 15])
             .build())

crossval = CrossValidator(estimator=rf_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator_auc,
                          numFolds=3)

cv_model = crossval.fit(train)
best_rf_model = cv_model.bestModel

# Avaliar o modelo otimizado
best_predictions = best_rf_model.transform(test)
best_auc = evaluator_auc.evaluate(best_predictions)
print(f"Melhor Random Forest (AUC): {best_auc:.4f}")

# COMMAND ----------


## 8. Visualização dos resultados

# Gráfico de comparação
metrics = ['AUC', 'AUC-PR']
lr_scores = [lr_auc, lr_pr]
rf_scores = [rf_auc, rf_pr]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, lr_scores, width, label='Regressão Logística')
rects2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest')

ax.set_ylabel('Scores')
ax.set_title('Comparação de Modelos por Métrica')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

fig.tight_layout()
plt.show()

# COMMAND ----------

"""
Os modelos não estão se saindo bem, estão um pouquinho melhor que um chute, por isso seria importante testar outras técnicas de balanceamento de classes, já que a variável target é rara (26%).
As técnicas podem ser: SMOTE: criar dados sintéticos com a classe minoritária
                       Undersampling: remover dados da classe majoritária

Comentário adicional: o modelo XGBoost pode funcionar bem porque ajusta erros iterativamente e funciona bem com muitas variáveis.
"""