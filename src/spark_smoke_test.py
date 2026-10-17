"""
V5 - Apache Spark Smoke Test

Purpose:
    Verify that Python, PySpark, Java, and local Spark execution
    are configured correctly before running distributed embedding
    experiments.

This test:
    1. Creates a local SparkSession.
    2. Creates a small distributed dataset.
    3. Executes a parallel transformation.
    4. Executes an aggregation.
    5. Prints Spark configuration information.
"""

from __future__ import annotations

import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():

    print()
    print("V5 - Apache Spark Smoke Test")
    print("============================")
    print()

    print(f"Python:  {sys.version.split()[0]}")
    print(f"PySpark: {pyspark.__version__}")

    # --------------------------------------------------------
    # Create local Spark session
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("ScalableRAGSpark-V5-SmokeTest")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print(f"Spark:   {spark.version}")
    print(
        f"Master:  "
        f"{spark.sparkContext.master}"
    )

    print(
        f"Default parallelism: "
        f"{spark.sparkContext.defaultParallelism}"
    )

    # --------------------------------------------------------
    # Create a small distributed dataset
    # --------------------------------------------------------

    rows = [
        (i,)
        for i in range(1, 101)
    ]

    df = spark.createDataFrame(
        rows,
        ["value"],
    )

    # Explicitly create four partitions.
    df = df.repartition(4)

    print()
    print(
        f"Partitions: "
        f"{df.rdd.getNumPartitions()}"
    )

    # --------------------------------------------------------
    # Distributed transformation
    # --------------------------------------------------------

    transformed = (
        df
        .withColumn(
            "square",
            F.col("value")
            * F.col("value"),
        )
    )

    # --------------------------------------------------------
    # Distributed aggregation
    # --------------------------------------------------------

    result = (
        transformed
        .agg(
            F.sum("square")
            .alias("sum_of_squares")
        )
        .collect()[0][
            "sum_of_squares"
        ]
    )

    expected = sum(
        i * i
        for i in range(1, 101)
    )

    print()
    print(
        f"Spark result:   {result}"
    )

    print(
        f"Expected result:{expected}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if result != expected:

        spark.stop()

        raise RuntimeError(
            "Spark smoke test failed: "
            "incorrect result."
        )

    print()
    print(
        "Spark smoke test PASSED."
    )

    print(
        "Local parallel Spark execution "
        "is working."
    )

    # --------------------------------------------------------
    # Stop Spark cleanly
    # --------------------------------------------------------

    spark.stop()


if __name__ == "__main__":
    main()