import sparknlp
from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

#create or get Spark Session

spark = sparknlp.start()

sparknlp.version()
spark.version

#download, load and annotate a text by pre-trained pipeline

pipeline = PretrainedPipeline('recognize_entities_dl', 'en')
result = pipeline.annotate('The Mona Lisa is a 16th century oil painting created by Leonardo')
print(result)