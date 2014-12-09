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

package org.apache.spark.mllib.tree

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.MyLabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.MyBoostingStrategy
import org.apache.spark.mllib.tree.impl.{MyTreePoint, TimeTracker}
import org.apache.spark.mllib.tree.impurity.{MyVariance, Variance}
import org.apache.spark.mllib.tree.model.{Bin, Split, MyDecisionTreeModel, MyGradientBoostedTreesModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * :: Experimental ::
 * A class that implements
 * [[http://en.wikipedia.org/wiki/Gradient_boosting  Stochastic Gradient Boosting]]
 * for regression and binary classification.
 *
 * The implementation is based upon:
 *   J.H. Friedman.  "Stochastic Gradient Boosting."  1999.
 *
 * Notes on Gradient Boosting vs. TreeBoost:
 *  - This implementation is for Stochastic Gradient Boosting, not for TreeBoost.
 *  - Both algorithms learn tree ensembles by minimizing loss functions.
 *  - TreeBoost (Friedman, 1999) additionally modifies the outputs at tree leaf nodes
 *    based on the loss function, whereas the original gradient boosting method does not.
 *     - When the loss is SquaredError, these methods give the same result, but they could differ
 *       for other loss functions.
 *
 * @param boostingStrategy Parameters for the gradient boosting algorithm.
 */
@Experimental
class MyGradientBoostedTrees(private val boostingStrategy: MyBoostingStrategy)
  extends Serializable with Logging {

  /**
   * Method to train a gradient boosting model
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.MyLabeledPoint]].
   * @return a gradient boosted trees model that can be used for prediction
   */
  def run(input: RDD[MyLabeledPoint]): MyGradientBoostedTreesModel = {
    ???
    /*
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Regression => MyGradientBoostedTrees.boost(input, boostingStrategy)
      case Classification =>

        // Map labels to -1, +1 so binary classification can be treated as regression.
        val remappedInput = input.map(x => new MyLabeledPoint((x.label * 2) - 1, x.features))
        MyGradientBoostedTrees.boost(remappedInput, boostingStrategy)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the gradient boosting.")
    }*/
  }

  def run(
          input: RDD[MyTreePoint],
          boostingStrategy: MyBoostingStrategy,
          numFeatures:Int,
          numExamples:Int,
          splits:Array[Array[Split]],
          bins:Array[Array[Bin]],
          featureSubsetStrategy:String = "all") = {
    MyGradientBoostedTrees.boost(input, boostingStrategy, numFeatures, numExamples, splits, bins, featureSubsetStrategy)
  }


  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.MyGradientBoostedTrees!#run]].
   */
  def run(input: JavaRDD[MyLabeledPoint]): MyGradientBoostedTreesModel = {
    run(input.rdd)
  }
}


object MyGradientBoostedTrees extends Logging {

  /**
   * Method to train a gradient boosting model.
   *
   * @param input Training dataset: RDD of [[org.apache.spark.mllib.regression.MyLabeledPoint]].
   *              For classification, labels should take values {0, 1, ..., numClasses-1}.
   *              For regression, labels are real numbers.
   * @param boostingStrategy Configuration options for the boosting algorithm.
   * @return a gradient boosted trees model that can be used for prediction
   */
  def train(
      input: RDD[MyLabeledPoint],
      boostingStrategy: MyBoostingStrategy): MyGradientBoostedTreesModel = {
    new MyGradientBoostedTrees(boostingStrategy).run(input)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.GradientBoostedTrees#train]]
   */
  def train(
      input: JavaRDD[MyLabeledPoint],
      boostingStrategy: MyBoostingStrategy): MyGradientBoostedTreesModel = {
    train(input.rdd, boostingStrategy)
  }

  /**
   * Internal method for performing regression using trees as base learners.
   * @param input training dataset
   * @param boostingStrategy boosting parameters
   * @return a gradient boosted trees model that can be used for prediction
   */
  private def boost(
      input: RDD[MyTreePoint],
      boostingStrategy: MyBoostingStrategy,
      numFeatures:Int,
      numExamples:Int,
      splits:Array[Array[Split]],
      bins:Array[Array[Bin]],
      featureSubsetStrategy:String = "all"): MyGradientBoostedTreesModel = {

    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numIterations = boostingStrategy.numIterations
    val baseLearners = new Array[MyDecisionTreeModel](numIterations)
    val baseLearnerWeights = new Array[Double](numIterations)
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    treeStrategy.algo = Regression
    treeStrategy.impurity = MyVariance
    treeStrategy.assertValid()

    // Cache input
    if (input.getStorageLevel == StorageLevel.NONE) {
      //input.persist(StorageLevel.MEMORY_AND_DISK) todo WORK OUT WHY I CAN'T SAVE IN MEMORY!
    }

    timer.stop("init")

    logDebug("##########")
    logDebug("Building tree 0")
    logDebug("##########")
    var data = input

    // Initialize tree
    timer.start("building tree 0")
    val firstTreeModel = new MyDecisionTree(treeStrategy).run(data, numFeatures, numExamples, splits, bins, featureSubsetStrategy)
    baseLearners(0) = firstTreeModel
    baseLearnerWeights(0) = 1.0
    val startingModel = new MyGradientBoostedTreesModel(Regression, Array(firstTreeModel), Array(1.0))
    logDebug("error of gbt = " + loss.computeError(startingModel, input))
    // Note: A model of type regression is used since we require raw prediction
    timer.stop("building tree 0")

    // psuedo-residual for second iteration
    data = loss.gradient(startingModel, input).map{ case (p, grad) => {
      p.copy(label = grad)
    }}

    var m = 1
    while (m < numIterations) {
      timer.start(s"building tree $m")
      logDebug("###################################################")
      logDebug("Gradient boosting tree iteration " + m)
      logDebug("###################################################")
      val model = new MyDecisionTree(treeStrategy).run(data, numFeatures, numExamples, splits, bins, featureSubsetStrategy)
      timer.stop(s"building tree $m")
      // Create partial model
      baseLearners(m) = model
      // Note: The setting of baseLearnerWeights is incorrect for losses other than SquaredError.
      //       Technically, the weight should be optimized for the particular loss.
      //       However, the behavior should be reasonable, though not optimal.
      baseLearnerWeights(m) = learningRate
      // Note: A model of type regression is used since we require raw prediction
      val partialModel = new MyGradientBoostedTreesModel(
        Regression, baseLearners.slice(0, m + 1), baseLearnerWeights.slice(0, m + 1))
      logDebug("error of gbt = " + loss.computeError(partialModel, input))
      // Update data with pseudo-residuals
//      data = input.map(point => MyLabeledPoint(loss.gradient(partialModel, point) * -1,
//        point.features))
      data = loss.gradient(partialModel, input).map{ case (p, grad) => {
        p.copy(label = grad)
      }}
      m += 1
    }

    timer.stop("total")

    logInfo("Internal timing for MyDecisionTree:")
    logInfo(s"$timer")

    new MyGradientBoostedTreesModel(
      boostingStrategy.treeStrategy.algo, baseLearners, baseLearnerWeights)
  }
}
