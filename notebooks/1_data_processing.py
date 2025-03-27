# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
from pyspark.ml.stat import Correlation
from pyspark.sql.types import DoubleType, IntegerType, LongType

# COMMAND ----------

# Criar sessão Spark
spark = SparkSession.builder.appName("EDA_Analysis").getOrCreate()

# COMMAND ----------

# Carregar os dados
offers_df = spark.read.json("/data/raw/offers.json")
customers_df = spark.read.json("/data/raw/profile.json")
transactions_df = spark.read.json("/data/raw/transactions.json")

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA

# COMMAND ----------

# Verificar schema e primeiras linhas
offers_df.printSchema()
offers_df.show(5)

# COMMAND ----------

customers_df.printSchema()
customers_df.show(5)

# COMMAND ----------

transactions_df.printSchema()
transactions_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estatísticas descritivas

# COMMAND ----------

# Estatísticas Descritivas sobre os clientes
print("Estatísticas de clientes:")
customers_df.describe(["age", "credit_card_limit"]).show()

# COMMAND ----------

# Estatísticas Descritivas sobre as ofertas
print("Estatísticas de ofertas:")
offers_df.describe(["min_value", "duration", "discount_value"]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise dos Clientes

# COMMAND ----------

# Rever as primeiras linhas
customers_df.show(5)

# COMMAND ----------

# Ver quais as categorias existem na coluna "gender"
customers_df.groupBy("gender").count().toPandas()

# COMMAND ----------

# Distribuição de gênero dos clientes
"""
Como vimos, existem dados vazios e dados marcados como "O", precisamos tratar esses dados para os gêneros F/M e desconhecido
"""

# Substituir valores nulos na coluna 'gender' por "Desconhecido"
customers_df = customers_df.withColumn(
    "gender",
    when((col("gender") == "") | (col("gender") == "O") | col("gender").isNull(), "genero_desconhecido")
    .when(col("gender") == "F", "genero_feminino")
    .when(col("gender") == "M", "genero_masculino")
)

# Agrupar os dados por gênero e contar a quantidade de registros
gender_count = customers_df.groupBy("gender").count().toPandas()

# Criar o gráfico de pizza
plt.figure(figsize=(8, 8))  # Define o tamanho da figura
plt.pie(
    gender_count['count'],  # Valores para cada fatia
    labels=gender_count['gender'],  # Rótulos para cada fatia
    autopct='%1.1f%%',  # Mostra os percentuais com uma casa decimal
    startangle=90,  # Inicia o gráfico no ângulo 90 graus (meio-dia)
    colors=sns.color_palette('pastel'),  # Usa uma paleta de cores suaves
)

# Adicionar título
plt.title("Distribuição de Gênero")

# Mostrar o gráfico
plt.show()

# COMMAND ----------

# Distribuição da idade dos clientes
customers_pd = customers_df.select("age").toPandas()
sns.histplot(customers_pd["age"], bins=30, kde=True)
plt.title("Distribuição de Idade dos Clientes")
plt.xlabel("Idade")
plt.ylabel("Frequência")
plt.show()

# COMMAND ----------

# Contar as idades
"""
Como é possível ver, existem registro que as idades são maiores de cem anos, aparentemente são a maioria dos registros, isso é estranho, uma vez que estamos falando em idade, não é comum ter pessoas com mais de cem anos e ainda mais usando a tecnologia.
Por isso queremos ver quantos registros estão nessa faixa etária.
"""

# Calcular a contagem de cada idade
age_counts = customers_df.groupBy("age").count()

# Calcular o total de registros
total_count = age_counts.select(sum(col("count"))).collect()[0][0]

# Adicionar a coluna de percentual
age_counts_with_percent = age_counts.withColumn(
    "percentual",
    round((col("count") / total_count * 100), 2)  # Arredonda para 2 casas decimais
)
# Mostrar o resultado
age_counts_with_percent.orderBy(col("count").desc()).show(5)  # Ordenar por contagem (decrescente)

# COMMAND ----------

# Distribuição dos limites de crédito
credit_count = customers_df.select("credit_card_limit").toPandas()
sns.histplot(credit_count["credit_card_limit"], bins=30, kde=True)
plt.title("Distribuição de Limites de Crédito")
plt.xlabel("Limite")
plt.ylabel("Frequência")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise das Ofertas

# COMMAND ----------

# Rever as 5 primeiras linhas
offers_df.show(5)

# COMMAND ----------

# Tipo de oferta
"""
É possível ver que exite uma coluna chamada offer_type, queremos ver quais são os tipos de ofertas
"""

# Contagem de tipos de oferta
offers_count = offers_df.groupBy("offer_type").count().toPandas()

# Calcular os percentuais
total = offers_count['count'].sum()
offers_count['percentual'] = (offers_count['count'] / total) * 100

# Criar o gráfico de barras
sns.barplot(data=offers_count, x="offer_type", y="count")

# Adicionar os percentuais no topo das barras
for i, row in offers_count.iterrows():
    plt.text(i, row['count'] + 0.1, f'{row["percentual"]:.1f}%', ha='center', va='bottom')

offers_count = offers_df.groupBy("offer_type").count().toPandas()

# Adicionar título e rótulos
plt.title("Quantidade de Ofertas por Tipo")
plt.xlabel("Tipo de Oferta")
plt.ylabel("Quantidade")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise das Transações

# COMMAND ----------

# Rever as 5 primeiras linhas
transactions_df.show(5)

# COMMAND ----------

# Rever schema
transactions_df.printSchema()

# COMMAND ----------

# Coluna value possui dados compactados
transactions_df.select("value").show(5, truncate=False)

# COMMAND ----------

# Descompactando a coluna "value"
"""
Precisamos "descompactar" os dados da coluna "value" para conseguirmos ver com clareza cada dado
"""

# Expandir a coluna 'value' em colunas separadas
df_expanded = transactions_df.select(
    "*",  # Seleciona todas as colunas existentes
    col("`value`.`offer id`").alias("offer_id_temp"),  # Renomeia para evitar conflito
    col("value.offer_id").alias("offer_id_temp2"),  # Renomeia para evitar conflito
    col("value.reward"),
    col("value.amount")
).drop("value")  # Remove a coluna 'value' original

# Consolidar as colunas 'offer id' e 'offer_id' em uma única coluna 'offer_id'
transactions_df = df_expanded.withColumn(
    "offer_id",  # Nome da nova coluna consolidada
    coalesce(col("offer_id_temp"), col("offer_id_temp2"))  # Usa o primeiro valor não nulo
).drop("offer_id_temp", "offer_id_temp2")  # Remove as colunas temporárias

# Mostrar o resultado
transactions_df.show(5)

# COMMAND ----------

# Os eventos podem ser diferentes
"""
Aparentemente a coluna "event" possui mais de uma categoria, vamos ver quais e quantas são
"""

# Contagem de eventos nas transações
events_count = transactions_df.groupBy("event").count().toPandas()

# Calcular os percentuais
total = events_count['count'].sum()
events_count['percentual'] = (events_count['count'] / total) * 100

# Criar o gráfico de barras
sns.barplot(data=events_count, x="event", y="count")

# Adicionar os percentuais no topo das barras
for i, row in events_count.iterrows():
    plt.text(i, row['count'] + 0.1, f'{row["percentual"]:.1f}%', ha='center', va='bottom')

# Adicionar título e rótulos
plt.title("Quantidade de Eventos de Transação")
plt.xlabel("Evento")
plt.ylabel("Quantidade")
plt.xticks(rotation=45)

# Mostrar o gráfico
plt.show()

# COMMAND ----------

# Contar as transações que tiveram ofertas
"""
Dá para ver que parte das transações não receberam ofertas, queremos ver quantas transações receberam e quantas não receberam ofertas
"""

# Criar coluna binária
transactions_df = transactions_df.withColumn("received_offer", 
                   when(col("event") == "transaction", 0).otherwise(1))

# Contar a quantidade de 0s e 1s
counts_df = transactions_df.groupBy("received_offer").count()
total = transactions_df.count()
counts = counts_df.collect()

# Preparar os dados para o gráfico
labels = ['0: Não recebeu oferta', '1: Recebeu oferta']
values = [0, 0]
percentages = [0, 0]

for row in counts:
    if row['received_offer'] == 0:
        values[0] = row['count']
        percentages[0] = (row['count'] / total) * 100
    else:
        values[1] = row['count']
        percentages[1] = (row['count'] / total) * 100

# Criar o gráfico de barras
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=['skyblue', 'lightgreen'])

# Adicionar título e rótulos
plt.title('Distribuição de Ofertas Recebidas')
plt.xlabel('Categoria')
plt.ylabel('Quantidade de Transações')

# Adicionar os valores absolutos e percentuais em cima de cada barra
for bar, value, percentage in zip(bars, values, percentages):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value}\n({percentage:.1f}%)',
             ha='center', va='bottom')

# Mostrar o gráfico
plt.tight_layout()  # Ajusta o layout para evitar cortes
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Transformações

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tratamentos - tabela clientes

# COMMAND ----------

# Faixa etária
"""
Ao invés de lidarmos com as idades como numeros inteiros, vamos criar faixas etárias, assim conseguimos reduzir a quantidade de variações
"""

# Criar variável categorica da faixa etária
customers_df_trated = customers_df.withColumn(
    "age_range",
    when(col("age") < 20, "menor_20_anos_idade")
     .when((col("age") >= 20) & (col("age") < 30), "20-29_anos_idade")
     .when((col("age") >= 30) & (col("age") < 40), "30-39_anos_idade")
     .when((col("age") >= 40) & (col("age") < 50), "40-49_anos_idade")
     .when((col("age") >= 50) & (col("age") < 60), "50-59_anos_idade")
     .when((col("age") >= 60) & (col("age") < 70), "60-69_anos_idade")
     .when((col("age") >= 70) & (col("age") < 80), "70-79_anos_idade")
     .when((col("age") >= 80) & (col("age") < 90), "80-89_anos_idade")
     .when((col("age") >= 90) & (col("age") < 100), "90-99_anos_idade")
     .otherwise("mais_100_anos_idade")
)

# Visualizar resultados
customers_df_trated.select("age", "age_range").show(5)

# COMMAND ----------

# Agrupar e contar e Mostrar os resultados
customers_df_trated.groupBy("age_range").count().orderBy("age_range").show()

# COMMAND ----------

# Calcular o tempo de cadastro
"""
Queremos saber o tempo desde o cadastro, pode ser que clientes mais novos sejam mais engajados ou o iverso, para isso precisamos saber se o tempo de cadastro é um fator importante na decisão de receber e aceitar uma oferta
"""

# Transformar a coluna registered_on em data
customers_df_trated = customers_df_trated.withColumn(
    "registration_date", 
    to_date(col("registered_on").cast("string"), "yyyyMMdd")
)

# Calcular o tempo de resgitro em anos
customers_df_trated = customers_df_trated.withColumn(
    "years_since_registration",
    floor(months_between(current_date(), col("registration_date")) / 12)
)

# Criar variável categorica do tempo desde o registro em anos
customers_df_trated = customers_df_trated.withColumn(
    "registration_years_category",
    when(col("years_since_registration") < 1, "menos_1_ano_cadastro")
     .when((col("years_since_registration") >= 1) & (col("years_since_registration") < 3), "1-2_anos_cadastro")
     .when((col("years_since_registration") >= 3) & (col("years_since_registration") < 5), "3-4_anos_cadastro")
     .when((col("years_since_registration") >= 5) & (col("years_since_registration") < 7), "5-6_anos_cadastro")
     .when((col("years_since_registration") >= 7) & (col("years_since_registration") < 9), "7-8_anos_cadastro")
     .when((col("years_since_registration") >= 9) & (col("years_since_registration") < 11), "9-10_anos-cadastro")
     .when((col("years_since_registration") >= 11) & (col("years_since_registration") < 13), "11-12_anos_cadastro")
     .when((col("years_since_registration") >= 13) & (col("years_since_registration") < 15), "13-14_anos_cadastro")
     .when((col("years_since_registration") >= 15) & (col("years_since_registration") < 17), "15-16_anos_cadastro")
     .when((col("years_since_registration") >= 17) & (col("years_since_registration") < 19), "17-18_anos_cadastro")
     .otherwise("mais_18_anos_cadastro")
)\
.drop(
    "years_since_registration", "registered_on"
)

# Visualizar resultados
customers_df_trated.select("registration_date", "registration_years_category",
          ).show(5)

# Agrupar categorias
customers_df_trated.groupBy("registration_years_category").count().show()

# COMMAND ----------

# Limite de crédito
"""
Vimos que existem várias faixas de limite de crédito, vamos criar categorias para conseguirmos reduzir a variedade, mas vimos que existem dados nulos que precisarão ser incluídos em alguma categoria para não "inflar" categorias erradas
"""

# Tratar variáveis com valores nulos
customers_df_trated = customers_df_trated.withColumn(
    "credit_card_limit",
    when((col("credit_card_limit") == "") | (col("credit_card_limit") == "O") | col("credit_card_limit").isNull(), "0")  # Trata vazios e zeros
    .otherwise(col("credit_card_limit"))  # Mantém os valores originais
)

# Criar variáveis categóricas da faixa de limite do cartão
customers_df_trated = customers_df_trated.withColumn(
    "credit_limit_range",
    when(col("credit_card_limit") < 1, "limite_desconhecido")
     .when((col("credit_card_limit") >= 1) & (col("credit_card_limit") < 10000), "menos_10_mil_limite")
     .when((col("credit_card_limit") >= 10000) & (col("credit_card_limit") < 30000), "10-20_mil_limite")
     .when((col("credit_card_limit") >= 30000) & (col("credit_card_limit") < 50000), "30-40_mil_limite")
     .when((col("credit_card_limit") >= 50000) & (col("credit_card_limit") < 70000), "50-60_mil_limite")
     .when((col("credit_card_limit") >= 70000) & (col("credit_card_limit") < 90000), "70-80_mil_limite")
     .when((col("credit_card_limit") >= 90000) & (col("credit_card_limit") < 110000), "90-100_mil_limite")
     .when((col("credit_card_limit") >= 110000) & (col("credit_card_limit") < 130000), "110-120_mil_limite")
     .when((col("credit_card_limit") >= 130000) & (col("credit_card_limit") < 150000), "130-140_mil_limite")
     .when((col("credit_card_limit") >= 150000) & (col("credit_card_limit") < 170000), "150-160_mil_limite")
     .when((col("credit_card_limit") >= 170000) & (col("credit_card_limit") < 190000), "170-180_mil_limite")
     .otherwise("mais_180_mil")
)

# Visualizar resultados
customers_df_trated.select("credit_card_limit", "credit_limit_range",
          ).show(5)

# Agrupar categorias
customers_df_trated.groupBy("credit_limit_range").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tratamentos - Tabela de ofertas

# COMMAND ----------

# Rever a tabela
offers_df.show(5)

# COMMAND ----------

# Canal de distribuição da oferta
"""
Os canais estão compactados numa mesma coluna, para saber se existem canais mais importantes que outros, precisaremos separá-los. Como a mesma oferta pode ser veiculada por mais de um canal, optei por fazer a técnica "one-hot-encoder" que é separar os canais e marcar com zeros e um, sendo 1 onde a oferta é veiculada e 0 nos canais que a oferta não será veiculada.
"""

# Explodir a coluna 'channels' para ter uma linha por canal
exploded_df = offers_df.withColumn("channel", explode(col("channels")))

# Criar colunas binárias para cada canal
binary_df = exploded_df.groupBy("id").pivot("channel").count()

# Preencher valores nulos com 0 e substituir contagens maiores que 0 por 1
for column in binary_df.columns:
    if column != "id":
        binary_df = binary_df.withColumn(column, when(col(column) > 0, 1).otherwise(0))

# Juntar o DataFrame original com as colunas binárias
offers_exploded = offers_df.join(binary_df, on="id", how="left")

# Mostrar o resultado
offers_exploded.show(truncate=False)

# COMMAND ----------

# Contar e somar
"""
Já que a mesma oferta pode ser veiculada em mais de um canal, fiquei curiosa sobre: existem ofertas que estarão em todos os canais? Quais canais recebem mais ofertas?
"""

# Somar as colunas (quantas ofertas cada canal tem) 
channel_sums = offers_exploded.select(
    *[sum(col(channel)).alias(channel) for channel in binary_df.columns if channel != "id"]
)

# Mostrar a soma das colunas
print("Soma das colunas (ofertas por canal):")
channel_sums.show(vertical=True)

# Somar as linhas (quantos canais cada oferta está presente)
# Criar uma expressão SQL para somar as colunas dinamicamente
sum_expression = " + ".join([channel for channel in binary_df.columns if channel != "id"])

# Adicionar a soma das colunas como uma nova coluna
offers_exploded_sum = offers_exploded.withColumn("total_channels", expr(sum_expression))

# Mostrar o DataFrame com a soma das linhas
print("DataFrame com a soma das linhas (canais por oferta):")
offers_exploded_sum.select("id", "offer_type", "total_channels").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tratamentos - Tabela de transações

# COMMAND ----------

# Rever as 5 primeiras linhas
transactions_df.show(5)

# COMMAND ----------

# Tempo desde o começo do teste
transactions_df.groupBy("time_since_test_start").count().show()

# COMMAND ----------

# Tempo desde o teste
"""
Vimos que existem diferentes tempos (dias) desde o começo do teste, vamos criar faixas para categorizar esses dados e diminuir a sua variabilidade
"""

# Criar variáveis categóricas da faixa de limite do cartão
transactions_df = transactions_df.withColumn(
    "time_since_test_start_range",
    when(col("time_since_test_start") < 1, "menos_um_dia_test")
     .when((col("time_since_test_start") >= 1) & (col("time_since_test_start") < 5), "1-4_dias_test")
     .when((col("time_since_test_start") >= 5) & (col("time_since_test_start") < 10), "5-9_dias_test")
     .when((col("time_since_test_start") >= 10) & (col("time_since_test_start") < 15), "10-14_dias_test")
     .when((col("time_since_test_start") >= 15) & (col("time_since_test_start") < 20), "15-19_dias_test")
     .when((col("time_since_test_start") >= 20) & (col("time_since_test_start") < 29), "20-29_dias_test")
     .otherwise("mais_30_dias_test")
)

# Visualizar resultados
transactions_df.select("time_since_test_start", "time_since_test_start_range",
          ).show()

# Agrupar categorias
transactions_df.groupBy("time_since_test_start_range").count().show()

# COMMAND ----------

# Transações com ofertas
"""
Como precisamos entender quais as ofertas foram mais atrativas, ou seja, que foram aceitas, vamos recortar apenas as transações que receberam ofertas, assim, conseguimos analisar quais ofertas foram atraentes e quais não foram.
"""

# Recortar dataset
transactions_offers = transactions_df.filter("received_offer == 1")

# Visualizar resultado
transactions_offers.groupBy("received_offer").count().show()
transactions_offers = transactions_offers.drop("received_offer")
transactions_offers.show(5)

# COMMAND ----------

offers_exploded_sum.show(5)

# COMMAND ----------

# Merge dados de transação com ofertas
transactions_offers = transactions_offers.join(
    offers_exploded_sum, transactions_offers.offer_id == offers_exploded_sum.id, how="left"
        )\
        .drop("id", "received_offer", "channels")

# Visualizar resultados
display(transactions_offers.limit(5).toPandas())

# COMMAND ----------

# Rever 5 primeiras linhas da tabela de clientes
display(customers_df_trated.limit(5))

# COMMAND ----------

# Merge com os dados dos clientes
transactions_offers_clientes = transactions_offers.join(
    customers_df_trated, 
    transactions_offers.account_id == customers_df_trated.id
    )\
    .drop("id", "age", "credit_card_limit", "registration_date")

# Visualizar resultados
display(transactions_offers_clientes.limit(5))

# COMMAND ----------

len(transactions_offers_clientes.columns)

# COMMAND ----------

# Oferta e evento
"""
Vimos que temos evento de transação (que não tem uma oferta) e eventos que possuiram uma oferta, então dessas temos evento de recebimento de ofertas e evento de oferta vista e oferta concluída. 
Vamos analisar se todas as ofertas passaram por esses três eventos: recebimento, visualização e completude.
"""

# Passo 1: Lista de offer_id que têm pelo menos um evento "completed" (sem duplicatas)
completed_offers = transactions_offers_clientes.filter(col("event") == "offer completed") \
                    .select("offer_id", "event", "offer_type") \
                    .distinct()

# Passo 2: Lista de TODOS os offer_id únicos no dataset
all_offers = transactions_offers_clientes.select("offer_id", "event", "offer_type").distinct()

# Passo 3: Offer_id que NÃO estão na lista de completos (usando LEFT ANTI JOIN)
offers_nao_completadas = all_offers.join(
    completed_offers,
    on="offer_id",
    how="left_anti"  # Mantém apenas as linhas que NÃO têm match no join
)

# Mostrar o resultado
offers_nao_completadas.show()

# COMMAND ----------

# Dataset por tipo de oferta
"""
Vimos que as ofertas do tipo informacional não possuem o evento de completo, isso faz sentido porque o tipo é apenas informativo, sendo assim, vamos retirar esse tipo de oferta.
"""

# Recortar dataset
transactions_offers_clientes = transactions_offers_clientes.filter("offer_type != 'informational'")

# Ver resultado
transactions_offers_clientes.groupBy("offer_type").count().show()

# COMMAND ----------

# Visualizar as 5 primeiras linhas
display(transactions_offers_clientes.limit(5).toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dummy

# COMMAND ----------

# Transformar as variaveis categóricas em numéricas
"""
Anteriormente nós transformamos a idade, o limite do cartão e o tempo de cadastro em categorias, ou seja, criamos uma faixa etária, uma faixa de limites e uma faixa de anos de cadastro.
Porém, não podemos lidar com os dados da forma textuais, já que modelos aprendizado de máquina leem números, por isso vamos criar variáveis numéricas que consiste em: cada categoria virará uma coluna e será marcado com o valor 1 se aquele atributo estiver presente e 0 caso não esteja.
IMPORTANTE que seja feito de forma binária para não dar pesos e hierarquias às categorias.
"""

# Criar dummies das variáveis categóricas
dummie_genero = transactions_offers_clientes.groupBy("account_id").pivot("gender").agg(lit(1)).na.fill(0)
dummie_idade = transactions_offers_clientes.groupBy("account_id").pivot("age_range").agg(lit(1)).na.fill(0)
dummie_registro = transactions_offers_clientes.groupBy("account_id").pivot("registration_years_category").agg(lit(1)).na.fill(0)
dummie_limite = transactions_offers_clientes.groupBy("account_id").pivot("credit_limit_range").agg(lit(1)).na.fill(0)
dummie_teste = transactions_offers_clientes.groupBy("account_id").pivot("time_since_test_start_range").agg(lit(1)).na.fill(0)

# Visualizar resultados
dummie_genero.show(5)
dummie_idade.show(5)
dummie_registro.show(5)
dummie_limite.show(5)
dummie_teste.show(5)

# COMMAND ----------

# Unificar datasets
df_dummies = transactions_offers_clientes\
    .join(dummie_genero, "account_id", how="inner")\
    .join(dummie_idade, "account_id", how="inner")\
    .join(dummie_registro, "account_id", how="inner")\
    .join(dummie_limite, "account_id", how="inner")\
    .join(dummie_teste, "account_id", how="inner")\
        .drop("gender", "age_range", "registration_years_category", "credit_limit_range", "time_since_test_start_range")

# Visualizar as 5 primeiras linhas
display(df_dummies.limit(5))

# COMMAND ----------

# Variável target
df_dummies = df_dummies.withColumn(
    "target",
    when(col("event") == "offer completed", 1).otherwise(0))

# Ver resultado
df_dummies.groupBy("target").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Correlação

# COMMAND ----------

# Remover coluna
"""
A coluna reward está preenchida apenas nas linhas do target, então não faz sentido manter, ela iria enviesar os resultados
E a coluna amount está preenchida vazia quando o evento é "offer completed", vamos remover também
A coluna email está em todas as transações, não fará diferença, sendo assim, vamos remover também
"""
df_dummies = df_dummies.drop("reward", "amount", "email")

# Remover dados NA
try:
    df_dummies = df_dummies.fillna(0, subset=numeric_cols)
except:
    df_dummies = df_dummies.na.fill(0)

# Separar apenas dados numericos
numeric_cols = [field.name for field in df_dummies.schema.fields 
               if isinstance(field.dataType, (DoubleType, IntegerType, LongType))]

assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="features",
    handleInvalid="keep"
)
    
vector_df = assembler.transform(df_dummies).select("features")
corr_matrix = Correlation.corr(vector_df, "features").collect()[0][0]
corr_matrix = corr_matrix.toArray()
    
# Converter para Pandas apenas para visualização
corr_df = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
corr_df

# COMMAND ----------

# Encontrar o índice da coluna target
target_idx = numeric_cols.index('target')
target_corr = corr_df.iloc[:, target_idx].sort_values(ascending=False)
    
# Plotar
plt.figure(figsize=(10, 6))
target_corr.drop('target').plot(kind='barh', color='steelblue')
plt.title('Correlação com a variável Target', pad=20)
plt.xlabel('Coeficiente de Correlação')
plt.ylabel('Variáveis')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# Salvar (requer biblioteca delta-spark)
df_dummies.write.format("delta").mode("overwrite").save("/data/processed/df_processado.delta")