package org.apache.spark.mllib.tree.configuration

import java.io.{ObjectInputStream, FileInputStream}

import org.apache.spark.mllib.tree.model.MyEnsembleModelNew

/**
 * Created by luke on 16/04/15.
 */
trait InitialModel

case class InitialTrainModel(initialTrees:Int) extends InitialModel

case class InitialLoadedModel(location:String) extends InitialModel {
  def load() = {
    println("Loading Model " + location)
    val fis = new FileInputStream(location + "/model.txt")
    val ois = new ObjectInputStream(fis)
    val model = ois.readObject().asInstanceOf[MyEnsembleModelNew[_]]
    println("Model Loaded :)")
    model
  }
}
