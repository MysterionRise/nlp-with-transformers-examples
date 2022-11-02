from johnsnowlabs import *

#
# jsl.start()
jsl.install()
spark = sparknlp.start()

documentAssembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sparktokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

zero_shot_ner = finance.ZeroShotNerModel.pretrained("finner_roberta_zeroshot", "en", "finance/models") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("zero_shot_ner") \
    .setEntityDefinitions(
    {
        "DATE": ['When was the company acquisition?', 'When was the company purchase agreement?'],
        "ORG": ["Which company was acquired?"],
        "PRODUCT": ["Which product?"],
        "PROFIT_INCREASE": ["How much has the gross profit increased?"],
        "REVENUES_DECLINED": ["How much has the revenues declined?"],
        "OPERATING_LOSS_2020": ["Which was the operating loss in 2020"],
        "OPERATING_LOSS_2019": ["Which was the operating loss in 2019"]
    })

nerconverter = nlp.NerConverter() \
    .setInputCols(["document", "token", "zero_shot_ner"]) \
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    documentAssembler,
    sparktokenizer,
    zero_shot_ner,
    nerconverter,
]
)

sample_text = [
    "In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
    "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
    "While our gross profit margin increased to 81.4% in 2020 from 63.1% in 2019, our revenues declined approximately 27% in 2020 as compared to 2019."
    "We reported an operating loss of approximately $8,048,581 million in 2020 as compared to an operating loss of approximately $7,738,193 million in 2019."]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(
    F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias(
        "cols")) \
    .select(F.expr("cols['0']").alias("chunk"),
            F.expr("cols['3']['entity']").alias("ner_label")) \
    .filter("ner_label!='O'") \
    .show(truncate=False)
