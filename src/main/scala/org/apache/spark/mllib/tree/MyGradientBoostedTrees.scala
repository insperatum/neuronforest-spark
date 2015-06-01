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
import org.apache.spark.mllib.tree.configuration.{InitialLoadedModel, InitialTrainModel, MyBoostingStrategy}
import org.apache.spark.mllib.tree.impl.{MyTreePoint, TimeTracker}
import org.apache.spark.mllib.tree.impurity.{MyVariance, Variance}
import org.apache.spark.mllib.tree.loss.MalisLoss
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.tree.NeuronUtils.cached

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
  def run(input: RDD[MyLabeledPoint]): MyGradientBoostedTreesModelNew = {
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
         subsample_proportion: Double,
          featureSubsetStrategy:String,
          save_loss_to:String,
           save_gradients:Boolean) = {
    MyGradientBoostedTrees.boost(input, boostingStrategy, numFeatures, numExamples, splits, bins, subsample_proportion, featureSubsetStrategy, save_loss_to, save_gradients)
  }


  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.MyGradientBoostedTrees!#run]].
   */
  def run(input: JavaRDD[MyLabeledPoint]): MyGradientBoostedTreesModelNew = {
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
      boostingStrategy: MyBoostingStrategy): MyGradientBoostedTreesModelNew = {
    new MyGradientBoostedTrees(boostingStrategy).run(input)
  }

  /**
   * Java-friendly API for [[org.apache.spark.mllib.tree.GradientBoostedTrees#train]]
   */
  def train(
      input: JavaRDD[MyLabeledPoint],
      boostingStrategy: MyBoostingStrategy): MyGradientBoostedTreesModelNew = {
    train(input.rdd, boostingStrategy)
  }

  /**
   * Internal method for performing regression using trees as base learners.
   * @param input training dataset
   * @param boostingStrategy boosting parameters
   * @return a gradient boosted trees model that can be used for prediction
   */
  private def boost(
      uncachedInput: RDD[MyTreePoint],
      boostingStrategy: MyBoostingStrategy,
      numFeatures:Int,
      numExamples:Int,
      splits:Array[Array[Split]],
      bins:Array[Array[Bin]],
      subsample_proportion:Double,
      featureSubsetStrategy:String,
      save_loss_to:String,
      save_gradients:Boolean) = {

    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters

    val numIterations = boostingStrategy.numIterations
    val treesPerIteration = boostingStrategy.treesPerIteration
    val initialModel = boostingStrategy.initialModel
    val baseLearners = new Array[MyEnsembleModelNew[_]](numIterations + 1)
    val baseLearnerWeights = new Array[Double](numIterations + 1)
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    val momentum = boostingStrategy.momentum
    val losses = new Array[Double](numIterations + 1)
    val avgGradients = new Array[Double](numIterations + 1)
    // Prepare strategy for individual trees, which use regression with variance impurity.
    val forestStrategy = boostingStrategy.forestStrategy.copy
    forestStrategy.algo = Regression
    forestStrategy.impurity = MyVariance
    forestStrategy.assertValid()

    // Cache input
//    if (input.getStorageLevel == StorageLevel.NONE) {
//      //input.persist(StorageLevel.MEMORY_AND_DISK) //todo WORK OUT WHY I CAN'T SAVE IN MEMORY!
//    }
    val (input, unpersistInput) = cached(uncachedInput)

    timer.stop("init")

    println("##########")
    println(new java.util.Date)
    println("Building initial model")
    println("##########")
    var data = input

    // Initialize tree
    timer.start("building tree 0")
//    val firstTreeModel = new MyRandomForest(forestStrategy, treesPerIteration, featureSubsetStrategy, 1).
//      trainFromMyTreePoints(data, numFeatures, numExamples, splits, bins)
    val firstTreeModel = initialModel match {
      case m:InitialTrainModel => MyRandomForest.trainRegressorFromTreePoints(
        data, forestStrategy, m.initialTrees, featureSubsetStrategy, 1, numFeatures, numExamples, splits, bins)
      case m:InitialLoadedModel =>
        m.load()
    }

    baseLearners(0) = firstTreeModel
    baseLearnerWeights(0) = 1.0
    val startingModel = new MyGradientBoostedTreesModelNew(Regression, Array(firstTreeModel), Array(1.0))
    logDebug("error of gbt = " + loss.computeError(startingModel, input))
    // Note: A model of type regression is used since we require raw prediction
    timer.stop("building tree 0")

    // psuedo-residual for second iteration
    val (g_init, l_init, unc) = loss.cachedGradientAndLoss(startingModel, input, subsample_proportion, if(! save_gradients) null else save_loss_to + "/" + "gradient1")
    data = g_init.map{ case (p, grad) =>
      p.copy(label = grad)
    }
    //val (d, unc) = NeuronUtils.cached(data)
    println("Initial Loss = " + l_init)
    val g_init_avg = math.sqrt(g_init.map(_._2.sq).mean())
    println("Root Mean Square Gradient = " + g_init_avg)
    losses(0) = l_init
    avgGradients(0) = g_init_avg
    //data = d
    var uncacheData = unc

    import math.abs
    //data.map(d => (d.idx, d.label)).take(10).foreach(println)
    //println(data.map(d => abs(d.label._1) + abs(d.label._2) + abs(d.label._3)).reduce(_+_))
    //println(input.count())
    var m = 1
    while (m <= numIterations) {
      timer.start(s"building tree ${m}")
      println("###################################################")
      println(new java.util.Date)
      println("Gradient boosting tree iteration " + (m) + " of " + numIterations)
      println("###################################################")
      /*data.mapPartitionsWithIndex { (i, _) => {
        println("###################################################")
        println("Partition " + i + ", Gradient boosting tree iteration " + (m))
        println("###################################################")
        Iterator()
      }}.count()*/

      val model = MyRandomForest.trainRegressorFromTreePoints(data, forestStrategy, treesPerIteration, featureSubsetStrategy,
        1, numFeatures, numExamples, splits, bins)
      timer.stop(s"building tree ${m}")
      println("Finished building tree")
      // Create partial model
      baseLearners(m) = model
      // Note: The setting of baseLearnerWeights is incorrect for losses other than SquaredError.
      //       Technically, the weight should be optimized for the particular loss.
      //       However, the behavior should be reasonable, though not optimal.
      baseLearnerWeights(m) = learningRate
      // Note: A model of type regression is used since we require raw prediction

      println("Training partial model")
      val partialModel = new MyGradientBoostedTreesModelNew(
        Regression, baseLearners.slice(0, m+1), baseLearnerWeights.slice(0, m+1))
      logDebug("error of gbt = " + loss.computeError(partialModel, input))
      // Update data with pseudo-residuals
//      data = input.map(point => MyLabeledPoint(loss.gradient(partialModel, point) * -1,
//        point.features))

      m += 1

      uncacheData()
      //if(m <= numIterations) {
        println("Finding gradient and loss")
        val (g, l, unc2) = loss.cachedGradientAndLoss(partialModel, input, subsample_proportion,
          if(! save_gradients) null else save_loss_to + "/" + "gradient" + m
        )
        data = g.map { case (p, grad) =>
          p.copy(label = grad)
        }
        println("Loss = " + l)
        val g_avg = math.sqrt(g.map(_._2.sq).mean())
        println("Root Mean Square Gradient = " + g_avg)
        losses(m-1) = l
        avgGradients(m-1) = g_avg
        //val (d2, unc2) = NeuronUtils.cached(data)
        //data = d2
        uncacheData = unc2
      //}

      for(i <- 1 until m) {
        baseLearnerWeights(i) = baseLearnerWeights(i) + learningRate * math.pow(momentum, m-i)
      }
    }
    unpersistInput()
    timer.stop("total")

    logInfo("Internal timing for MyDecisionTree:")
    logInfo(s"$timer")

      NeuronUtils.saveText(save_loss_to, "malis.txt",
        "Losses = " + losses.map(_.toString).reduce(_ + ", " + _) +
          "\nGradients = " + avgGradients.map(_.toString).reduce(_ + ", " + _)
      )

//    (new MyGradientBoostedTreesModel(
//      boostingStrategy.forestStrategy.algo, baseLearners, baseLearnerWeights), data, seg)

    new MyGradientBoostedTreesModelNew(boostingStrategy.forestStrategy.algo, baseLearners, baseLearnerWeights)
  }
}
