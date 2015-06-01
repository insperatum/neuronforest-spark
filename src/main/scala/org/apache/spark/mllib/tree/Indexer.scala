package org.apache.spark.mllib.tree

class Indexer(val outerDimensions:(Int, Int), val minIdx:(Int, Int), val maxIdx:(Int, Int)) {
  val innerDimensions = (maxIdx._1 - minIdx._1 + 1, maxIdx._2 - minIdx._2 + 1)
  val outerSteps = (outerDimensions._2, 1)
  val innerSteps = (innerDimensions._2, 1)
  val size = innerDimensions._1 * innerDimensions._2

  def innerToOuter(i:Int) =
    outerSteps._1 * (minIdx._1 + i / innerSteps._1) +
      outerSteps._2 * (minIdx._2 + i % innerSteps._1)

  def outerToInner(i:Int) =
    innerSteps._1 * (i / outerSteps._1 - minIdx._1) +
      innerSteps._2 * (i % outerSteps._1 - minIdx._2)

  def innerToMulti(i:Int) =
    (i / innerSteps._1,
      (i % innerSteps._1) / innerSteps._2)
}