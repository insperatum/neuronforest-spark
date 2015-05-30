/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.impurity

import org.apache.spark.annotation.{DeveloperApi, Experimental}


/**
 * :: Experimental ::
 * Class for calculating variance during regression
 */
@Experimental
object MyVariance extends MyImpurity {

  /**
   * :: DeveloperApi ::
   * information calculation for multiclass classification
   * @param counts Array[Double] with counts for each label
   * @param totalCount sum of counts for all labels
   * @return information value, or 0 if totalCount = 0
   */
  @DeveloperApi
  override def calculate(counts: Array[Double], totalCount: Double): Double =
     throw new UnsupportedOperationException("MyVariance.calculate")

  /**
   * :: DeveloperApi ::
   * variance calculation
   * @param count number of instances
   * //@param sum sum of labels
   * //@param sumSquares summation of squares of the labels
   * @return information value, or 0 if count = 0
   */
  @DeveloperApi
  override def calculate(count: Double,
                         sum1: Double,
                         sumSquares1: Double): Double = {
    if (count == 0) {
      return 0
    }
    val squaredLoss = sumSquares1 - (sum1 * sum1) / count
    squaredLoss / count
  }

  /**
   * Get this impurity instance.
   * This is useful for passing impurity parameters to a Strategy in Java.
   */
  def instance = this

}

/**
 * Class for updating views of a vector of sufficient statistics,
 * in order to compute impurity from a sample.
 * Note: Instances of this class do not hold the data; they operate on views of the data.
 */
private[tree] class MyVarianceAggregator()
  extends MyImpurityAggregator(statsSize = 3) with Serializable {

  /**
   * Update stats for one (node, feature, bin) with the given label.
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def update(allStats: Array[Double], offset: Int, label: Double, instanceWeight: Double): Unit = {
    allStats(offset) += instanceWeight
    allStats(offset + 1) += instanceWeight * label
    allStats(offset + 2) += instanceWeight * label * label
  }

  /**
   * Get an [[MyImpurityCalculator]] for a (node, feature, bin).
   * @param allStats  Flat stats array, with stats for this (node, feature, bin) contiguous.
   * @param offset    Start index of stats for this (node, feature, bin).
   */
  def getCalculator(allStats: Array[Double], offset: Int): MyVarianceCalculator = {
    new MyVarianceCalculator(allStats.view(offset, offset + statsSize).toArray)
  }

}

/**
 * Stores statistics for one (node, feature, bin) for calculating impurity.
 * Unlike [[GiniAggregator]], this class stores its own data and is for a specific
 * (node, feature, bin).
 * @param stats  Array of sufficient statistics for a (node, feature, bin).
 */
private[tree] class MyVarianceCalculator(stats: Array[Double]) extends MyImpurityCalculator(stats) {

  require(stats.size == 3,
    s"MyVarianceCalculator requires sufficient statistics array stats to be of length 3," +
    s" but was given array of length ${stats.size}.")

  /**
   * Make a deep copy of this [[MyImpurityCalculator]].
   */
  def copy: MyVarianceCalculator = new MyVarianceCalculator(stats.clone())

  /**
   * Calculate the impurity from the stored sufficient statistics.
   */
  def calculate(): Double = MyVariance.calculate(stats(0), stats(1), stats(2))

  /**
   * Number of data points accounted for in the sufficient statistics.
   */
  def count: Long = stats(0).toLong

  /**
   * Prediction which should be made based on the sufficient statistics.
   */
  override def predict: Double = if (count == 0) {
    0.0
  } else {
    stats(1)/count
  }

  override def toString: String = {
    s"MyVarianceAggregator(cnt = ${stats(0)}, sum1 = ${stats(1)}, " +
    s"sumsq1 = ${stats(2)}"
  }

}
