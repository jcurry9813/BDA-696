import sys

from transform import rollingTransform
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


def main():

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    url = "jdbc:mysql://localhost:3306/baseball"

    gameq = """
            SELECT game_id, DATE(local_date) AS game_date
            FROM game
            """

    batter = """
            SELECT game_id, batter, Hit AS hit, atBat AS atbat
            FROM batter_counts WHERE atBat > 0
            """
# Enter your username and password to connect to baseball database

    user = "YourUserHere"
    password = "YourPassHere"

    game = (spark.read.format("jdbc").options(url=url, query=gameq, user=user, password=password).load())

    batter_counts = (spark.read.format("jdbc").options(url=url, query=batter, user=user, password=password).load())

    game.createOrReplaceTempView("game")
    batter_counts.createOrReplaceTempView("batter_counts")

    batter_join = batter_counts.join(game, on=["game_id"], how="left")

    game.unpersist()
    batter_counts.unpersist()
    batter_join.createOrReplaceTempView("batter_join")
    batter_join.persist(StorageLevel.MEMORY_ONLY)

    transform = rollingTransform(inputCols=["hit", "atbat"], outputCol="RollingBA")

    pipeline = Pipeline(stages=[transform])

    model = pipeline.fit(batter_join)
    batter_join = model.transform(batter_join)
    batter_join.show()
    return


if __name__ == "__main__":
    sys.exit(main())