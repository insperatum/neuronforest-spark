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

package org.apache.spark.mllib.tree.impl

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.MyLabeledPoint
import org.apache.spark.mllib.tree.{BinnedFeatureData, Double3}
import org.apache.spark.mllib.tree.model.Bin
import org.apache.spark.rdd.RDD


/**
 * Internal representation of MyLabeledPoint for DecisionTree.
 * This bins feature values based on a subsampled of data as follows:
 *  (a) Continuous features are binned into ranges.
 *  (b) Unordered categorical features are binned based on subsets of feature values.
 *      "Unordered categorical features" are categorical features with low arity used in
 *      multiclass classification.
 *  (c) Ordered categorical features are binned based on feature values.
 *      "Ordered categorical features" are categorical features with high arity,
 *      or any categorical feature used in regression or binary classification.
 */
case class MyTreePoint(label: Double3, seg:Int, data:BinnedFeatureData, idx:Int)
  extends Serializable {
  def binnedFeatures(f:Int) = {
    data.getBin(idx, f)
  }
  def features(f:Int) = data.getValue(idx, f)
  def getFeatureVector = Vectors.dense(Array.tabulate[Double](data.nFeatures)(features))
}


private[tree] object MyTreePoint {

  /**
   * Convert an input dataset into its MyTreePoint representation,
   * binning feature values in preparation for DecisionTree training.
   * @param input     Input dataset.
   * @param bins      Bins for features, of size (numFeatures, numBins).
   * @param metadata  Learning and dataset metadata
   * @return  MyTreePoint dataset representation
   */
  def convertToTreeRDD(
      input: RDD[MyLabeledPoint],
      bins: Array[Array[Bin]],
      metadata: MyDecisionTreeMetadata): RDD[MyTreePoint] = {
    // Construct arrays for featureArity and isUnordered for efficiency in the inner loop.
    val featureArity: Array[Int] = new Array[Int](metadata.numFeatures)
    val isUnordered: Array[Boolean] = new Array[Boolean](metadata.numFeatures)
    var featureIndex = 0
    while (featureIndex < metadata.numFeatures) {
      featureArity(featureIndex) = metadata.featureArity.getOrElse(featureIndex, 0)
      isUnordered(featureIndex) = metadata.isUnordered(featureIndex)
      featureIndex += 1
    }
    input.map { x =>
      MyTreePoint.labeledPointToTreePoint(x, bins, featureArity, isUnordered)
    }
  }

  /**
   * Convert one MyLabeledPoint into its MyTreePoint representation.
   * @param bins      Bins for features, of size (numFeatures, numBins).
   * @param featureArity  Array indexed by feature, with value 0 for continuous and numCategories
   *                      for categorical features.
   * @param isUnordered  Array index by feature, with value true for unordered categorical features.
   */
  private def labeledPointToTreePoint(
      labeledPoint: MyLabeledPoint,
      bins: Array[Array[Bin]],
      featureArity: Array[Int],
      isUnordered: Array[Boolean]): MyTreePoint = {
    val numFeatures = labeledPoint.features.size
    val arr = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      arr(featureIndex) = findBin(featureIndex, labeledPoint, featureArity(featureIndex),
        isUnordered(featureIndex), bins)
      featureIndex += 1
    }
    //new MyTreePoint(labeledPoint.label, arr)
    ???
  }

  /**
   * Find bin for one (labeledPoint, feature).
   *
   * @param featureArity  0 for continuous features; number of categories for categorical features.
   * @param isUnorderedFeature  (only applies if feature is categorical)
   * @param bins   Bins for features, of size (numFeatures, numBins).
   */
  private[tree] def findBin(
      featureIndex: Int,
      input: MyLabeledPoint,
      featureArity: Int,
      isUnorderedFeature: Boolean,
      bins: Array[Array[Bin]]): Int = {
    val binForFeatures = bins(featureIndex)
    val feature = input.features(featureIndex)
    findBin(feature, featureArity, isUnorderedFeature, binForFeatures)
  }

  private[tree] def findBin(
      featureValue: Double,
      featureArity: Int,
      isUnorderedFeature: Boolean,
      binForFeatures: Array[Bin]): Int = {
    /**
     * Binary search helper method for continuous feature.
     */
    def binarySearchForBins(): Int = {
      var left = 0
      var right = binForFeatures.length - 1
      while (left <= right) {
        val mid = left + (right - left) / 2
        val bin = binForFeatures(mid)
        val lowThreshold = bin.lowSplit.threshold
        val highThreshold = bin.highSplit.threshold
        if ((lowThreshold < featureValue) && (highThreshold >= featureValue)) {
          return mid
        } else if (lowThreshold >= featureValue) {
          right = mid - 1
        } else {
          left = mid + 1
        }
      }
      -1
    }

    if (featureArity == 0) {
      // Perform binary search for finding bin for continuous features.
      val binIndex = binarySearchForBins()
      if (binIndex == -1) {
        throw new RuntimeException("No bin was found for continuous feature." +
          " This error can occur when given invalid data values (such as NaN)." +
          " Feature index: $featureIndex.  Feature value: $featureValue")
      }
      binIndex
    } else {
      // Categorical feature bins are indexed by feature values.
      if (featureValue < 0 || featureValue >= featureArity) {
        throw new IllegalArgumentException(
          s"DecisionTree given invalid data:" +
            " Feature $featureIndex is categorical with values in" +
            " {0,...,${featureArity - 1}," +
            " but a data point gives it value $featureValue.\n" +
            "  Bad data point: ${features.toString}")
      }
      featureValue.toInt
    }
  }
}
