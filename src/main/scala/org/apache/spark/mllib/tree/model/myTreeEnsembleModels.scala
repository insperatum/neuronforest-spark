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

package org.apache.spark.mllib.tree.model

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import main.scala.org.apache.spark.mllib.tree.model.MyModel
import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.Double3
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.EnsembleCombiningStrategy._
import org.apache.spark.rdd.RDD

import scala.collection.mutable

/**
 * :: Experimental ::
 * Represents a random forest model.
 *
 * @param algo algorithm for the ensemble model, either Classification or Regression
 * @param trees tree ensembles
 */
@Experimental
class MyRandomForestModel(override val algo: Algo, val trees: Array[MyDecisionTreeModel])
  extends MyTreeEnsembleModel(algo, trees, Array.fill(trees.size)(1.0),
    combiningStrategy = Average) {

  require(trees.forall(_.algo == algo))
}

/**
 * :: Experimental ::
 * Represents a gradient boosted trees model.
 *
 * @param algo algorithm for the ensemble model, either Classification or Regression
 * @param elems tree ensembles
 * @param treeWeights tree ensemble weights
 */
@Experimental
class MyGradientBoostedTreesModel(
    override val algo: Algo,
    override val elems: Array[MyEnsembleModel[_]],
    override val treeWeights: Array[Double])
  extends MyEnsembleModel[MyEnsembleModel[_]](algo, elems, treeWeights, combiningStrategy = Sum) {

  require(elems.size == treeWeights.size)
}

/**
 * Represents a tree ensemble model.
 *
 * @param algo algorithm for the ensemble model, either Classification or Regression
 * @param elems tree ensembles
 * @param treeWeights tree ensemble weights
 * @param combiningStrategy strategy for combining the predictions, not used for regression.
 */
class MyEnsembleModel[T <: MyModel](
    protected val algo: Algo,
             val elems: Array[T],
    protected val treeWeights: Array[Double],
    protected val combiningStrategy: EnsembleCombiningStrategy) extends Serializable with MyModel {

  private val sumWeights = math.max(treeWeights.sum, 1e-15)

  def nElems = elems.size
  def getPartialModels:Seq[MyEnsembleModel[T]] = (1 to nElems).map(n =>
    new MyEnsembleModel(algo, elems.take(n), treeWeights.take(n), combiningStrategy))

  /**
   * Predicts for a single data point using the weighted sum of ensemble predictions.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  private def predictBySumming(features: Vector): Double3 = {
    val treePredictions = elems.map(_.predict(features))
    //blas.ddot(numTrees, treePredictions, 1, treeWeights, 1)
    treePredictions.zip(treeWeights).map {case (p, w) => p*w}.reduce(_+_)
  }

  /**
   * Classifies a single data point based on (weighted) majority votes.
   */
  private def predictByVoting(features: Vector): Double = {
    ???
    /*
    val votes = mutable.Map.empty[Int, Double]
    trees.view.zip(treeWeights).foreach { case (tree, weight) =>
      val prediction = tree.predict(features).toInt
      votes(prediction) = votes.getOrElse(prediction, 0.0) + weight
    }
    votes.maxBy(_._2)._1*/
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  def predict(features: Vector): Double3 = {
    (algo, combiningStrategy) match {
      case (Regression, Sum) =>
        predictBySumming(features)
      case (Regression, Average) =>
        predictBySumming(features) / sumWeights
      /*case (Classification, Sum) => // binary classification
        val prediction = predictBySumming(features)
        // TODO: predicted labels are +1 or -1 for GBT. Need a better way to store this info.
        if (prediction > 0.0) 1.0 else 0.0
      case (Classification, Vote) =>
        predictByVoting(features)
      */case _ =>
        throw new IllegalArgumentException(
          "MyTreeEnsembleModel given unsupported (algo, combiningStrategy) combination: " +
            s"($algo, $combiningStrategy).")
    }
  }

  /**
   * Predict values for the given data set.
   *
   * @param features RDD representing data points to be predicted
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predict(features: RDD[Vector]): RDD[Double3] = features.map(x => predict(x))

  /**
   * Java-friendly version of [[org.apache.spark.mllib.tree.model.MyEnsembleModel#predict]].
   */
  def predict(features: JavaRDD[Vector]): JavaRDD[java.lang.Double] = {
    predict(features.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Double]]
  }
}


private[tree] sealed class MyTreeEnsembleModel(
    algo: Algo,
    elems: Array[MyDecisionTreeModel],
    treeWeights: Array[Double],
    combiningStrategy: EnsembleCombiningStrategy)
  extends MyEnsembleModel[MyDecisionTreeModel](algo, elems, treeWeights, combiningStrategy) {

  /**
   * Print a summary of the model.
   */
  override def toString: String = {
    algo match {
      case Classification =>
        s"MyTreeEnsembleModel classifier with $numTrees trees\n"
      case Regression =>
        s"MyTreeEnsembleModel regressor with $numTrees trees\n"
      case _ => throw new IllegalArgumentException(
        s"MyTreeEnsembleModel given unknown algo parameter: $algo.")
    }
  }

  /**
   * Print the full model to a string.
   */
  def toDebugString: String = {
    val header = toString + "\n"
    header + elems.zipWithIndex.map { case (tree, treeIndex) =>
      s"  Tree $treeIndex:\n" + tree.topNode.subtreeToString(4)
    }.fold("")(_ + _)
  }

  /**
   * Get number of trees in forest.
   */
  def numTrees: Int = elems.size

  /**
   * Get total number of nodes, summed over all trees in the forest.
   */
  def totalNumNodes: Int = elems.map(_.numNodes).sum

  require(numTrees > 0, "MyTreeEnsembleModel cannot be created without trees.")
}














class MyEnsembleModelNew[T <: MyModel](
                                     val algo: Algo,
                                     val elems: Array[T],
                                     val treeWeights: Array[Double],
                                     val combiningStrategy: EnsembleCombiningStrategy) extends Serializable with MyModel {

  val sumWeights = math.max(treeWeights.sum, 1e-15)


  def isSum = combiningStrategy == Sum
  def nElems = elems.size
  def getPartialModels:Seq[MyEnsembleModelNew[T]] = (1 to nElems).map(n =>
    new MyEnsembleModelNew(algo, elems.take(n), treeWeights.take(n), combiningStrategy))

  def getPartialSegments(testPartialModels:Seq[Int]):Seq[MyEnsembleModelNew[_]] = {
    (0 +: testPartialModels.init).zip(testPartialModels).map{case (from, until) =>
      new MyEnsembleModelNew(algo, elems.drop(from).take(until-from), treeWeights.drop(from).take(until-from), combiningStrategy)
    }
  }
  /**
   * Predicts for a single data point using the weighted sum of ensemble predictions.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  private def predictBySumming(features: Vector): Double3 = {
    val treePredictions = elems.map(_.predict(features))
    //blas.ddot(numTrees, treePredictions, 1, treeWeights, 1)
    treePredictions.zip(treeWeights).map {case (p, w) => p*w}.reduce(_+_)
  }

  /**
   * Classifies a single data point based on (weighted) majority votes.
   */
  private def predictByVoting(features: Vector): Double = {
    ???
    /*
    val votes = mutable.Map.empty[Int, Double]
    trees.view.zip(treeWeights).foreach { case (tree, weight) =>
      val prediction = tree.predict(features).toInt
      votes(prediction) = votes.getOrElse(prediction, 0.0) + weight
    }
    votes.maxBy(_._2)._1*/
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return predicted category from the trained model
   */
  def predict(features: Vector): Double3 = {
    (algo, combiningStrategy) match {
      case (Regression, Sum) =>
        predictBySumming(features)
      case (Regression, Average) =>
        predictBySumming(features) / sumWeights
      /*case (Classification, Sum) => // binary classification
        val prediction = predictBySumming(features)
        // TODO: predicted labels are +1 or -1 for GBT. Need a better way to store this info.
        if (prediction > 0.0) 1.0 else 0.0
      case (Classification, Vote) =>
        predictByVoting(features)
      */case _ =>
        throw new IllegalArgumentException(
          "MyTreeEnsembleModel given unsupported (algo, combiningStrategy) combination: " +
            s"($algo, $combiningStrategy).")
    }
  }

  /**
   * Predict values for the given data set.
   *
   * @param features RDD representing data points to be predicted
   * @return RDD[Double] where each entry contains the corresponding prediction
   */
  def predict(features: RDD[Vector]): RDD[Double3] = features.map(x => predict(x))

  /**
   * Java-friendly version of [[org.apache.spark.mllib.tree.model.MyEnsembleModelNew#predict]].
   */
  def predict(features: JavaRDD[Vector]): JavaRDD[java.lang.Double] = {
    predict(features.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Double]]
  }
}



class MyTreeEnsembleModelNew(
                           algo: Algo,
                           elems: Array[MyDecisionTreeModel],
                           treeWeights: Array[Double],
                           combiningStrategy: EnsembleCombiningStrategy)
  extends MyEnsembleModelNew[MyDecisionTreeModel](algo, elems, treeWeights, combiningStrategy) {

  /**
   * Print a summary of the model.
   */
  override def toString: String = {
    algo match {
      case Classification =>
        s"MyTreeEnsembleModel classifier with $numTrees trees\n"
      case Regression =>
        s"MyTreeEnsembleModel regressor with $numTrees trees\n"
      case _ => throw new IllegalArgumentException(
        s"MyTreeEnsembleModel given unknown algo parameter: $algo.")
    }
  }

  /**
   * Print the full model to a string.
   */
  def toDebugString: String = {
    val header = toString + "\n"
    header + elems.zipWithIndex.map { case (tree, treeIndex) =>
      s"  Tree $treeIndex:\n" + tree.topNode.subtreeToString(4)
    }.fold("")(_ + _)
  }

  /**
   * Get number of trees in forest.
   */
  def numTrees: Int = elems.size

  /**
   * Get total number of nodes, summed over all trees in the forest.
   */
  def totalNumNodes: Int = elems.map(_.numNodes).sum

  require(numTrees > 0, "MyTreeEnsembleModel cannot be created without trees.")
}


@Experimental
class MyRandomForestModelNew(override val algo: Algo, val trees: Array[MyDecisionTreeModel])
  extends MyTreeEnsembleModelNew(algo, trees, Array.fill(trees.size)(1.0),
    combiningStrategy = Average) {

  require(trees.forall(_.algo == algo))
}


@Experimental
class MyGradientBoostedTreesModelNew(
                                   override val algo: Algo,
                                   override val elems: Array[MyEnsembleModelNew[_]],
                                   override val treeWeights: Array[Double])
  extends MyEnsembleModelNew[MyEnsembleModelNew[_]](algo, elems, treeWeights, combiningStrategy = Sum) {

  require(elems.size == treeWeights.size)
}