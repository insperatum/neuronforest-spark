package org.apache.spark.mllib.tree

import java.io._
import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.mllib.tree.configuration.{Algo, EnsembleCombiningStrategy}
import org.apache.spark.mllib.tree.model.{MyDecisionTreeModel, MyRandomForestModel, MyEnsembleModel}

/**
 * Created by luke on 17/04/15.
 */
object CombineForests {
  def main(args:Array[String]): Unit = {
    // Initial, no offsets
//    val save_to = "/masters_models/initial0"
//    val model_files = Array("/masters_models/2015-04-17 20-25-36", "/masters_models/2015-04-17 17-09-58")

      //Initial, offset 2
//    val save_to = "/masters_models/initial2"
//    val model_files = Array(
//      "2015-04-17 23-14-21",
//      "2015-04-17 23-15-25",
//      "2015-04-18 00-34-09",
//      "2015-04-18 00-53-06",
//      "2015-04-18 01-11-42",
//      "2015-04-18 01-30-37",
//      "2015-04-18 01-49-18",
//      "2015-04-18 02-08-04",
//      "2015-04-18 02-26-33",
//      "2015-04-18 02-45-02",
//      "2015-04-18 03-03-58",
//      "2015-04-18 03-23-01",
//      "2015-04-18 03-42-05",
//      "2015-04-18 04-02-14",
//      "2015-04-18 04-21-34",
//      "2015-04-18 04-40-32",
//      "2015-04-18 04-59-30",
//      "2015-04-18 05-18-20",
//      "2015-04-18 05-37-17",
//      "2015-04-18 05-56-10",
//      "2015-04-18 06-16-01",
//      "2015-04-18 06-35-00",
//      "2015-04-18 06-54-15",
//      "2015-04-18 07-13-20",
//      "2015-04-18 07-32-14",
//      "2015-04-18 07-51-20").map("/masters_models/" + _)

    val save_to = args(0) + "/" + new SimpleDateFormat("yyyy-MM-dd HH-mm-ss").format(new Date())
    val model_files = args.drop(1)


    new File(save_to).mkdirs()
    val fwdescription = new FileWriter(save_to + "/description.txt", false)


    val elems = model_files.map(file => {
      val fis = new FileInputStream(file + "/model.txt")
      val ois = new ObjectInputStream(fis)
      val model = ois.readObject().asInstanceOf[MyRandomForestModel]

      val desc = scala.io.Source.fromFile(file + "/description.txt").mkString
      fwdescription.write(desc + "\n--------------------------------\n")

      println("Loaded " + file)
      model
    }).flatMap(_.elems.toList)
    fwdescription.close()

    println("Creating Model")
    val combinedModel = new MyRandomForestModel(Algo.Regression, elems.toArray)

    println("Writing Model")
    val fos = new FileOutputStream(save_to + "/model.txt")
    val oos = new ObjectOutputStream(fos)
    oos.writeObject(combinedModel)
    oos.close()
  }
}
