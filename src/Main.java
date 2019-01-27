
/*
 GOAL: Load data from .arff files, preprocess the data, train a model and assess it either using 10-fold cross-validation or hold-out

 Compile:
 javac DataMiningExample.java

 Run Interpret:
 java DataMiningExample

 HACER!!!
	- Hacer modular
	- El programa no puede tener dependencias con datos!
	- Generar un .jar y ejecutar desde la línea de comandos

 */
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;


public class Main {

    public static void main(String[] args) throws Exception {
        /////////////////////////////////////////////////////////////
        // 1. LOAD DATA FILE
        //  HACER!!!! Bloque 1: como sub-clase
        // 1.1. Get the path of the .arff (instances) from the command line
        /*
		 if( args.length < 1 ){
			System.out.println("OBJETIVO: Seleccionar atributos (AttributeSelection<-CfsSubsetEval, search<-BestFirst) y Evaluar clasificador NaiveBayes con 10-fold cross-validation.");
			System.out.println("ARGUMENTOS:");
			System.out.println("\t1. Path del fichero de entrada: datos en formato .arff");
			return;
		}
		 */
        // 1.2. Open the file
        FileReader fi=null;
        try {
            fi= new FileReader("~/software/weka-3-6-9/data/breast-cancer.arff"); //(args[0]) <-> ("~/software/weka-3-6-9/data/breast-cancer.arff" )
        } catch (FileNotFoundException e) {
            System.out.println("ERROR: Revisar path del fichero de datos:"+args[0]);
        }
        // 1.3. Load the instances
        Instances data=null;
        try {
            data = new Instances(fi);
        } catch (IOException e) {
            System.out.println("ERROR: Revisar contenido del fichero de datos: "+args[0]);
        }
        // 1.4. Close the file
        try {
            fi.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
        }
        // 1.5. Shuffle the instances: apply Randomize filter
        //  HACER!!!!

        // 1.6. Specify which attribute will be used as the class: the last one, in this case
        data.setClassIndex(data.numAttributes()-1);

        /////////////////////////////////////////////////////////////
        // 2. FEATURE SUBSET SELECTION
        //  HACER!!!! Empaquetar Bloque 2: como sub-clase
        AttributeSelection filter= new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search=new BestFirst();
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);
        // 2.1 Get new data set with the attribute sub-set
        Instances newData = Filter.useFilter(data, filter);



        /////////////////////////////////////////////////////////////
        // 3. CLASSIFY:

        // 3.0 Train the classifier (estimador) by means of:	the Naive Bayes algorithm (in this case)
        NaiveBayes estimador= new NaiveBayes();//Naive Bayes

        // 3.1 Assess the performance of the classifier by means of 10-fold cross-validation
        //  HACER!!!! Empaquetar Bloque 3.1: como sub-clase
        Evaluation evaluator = new Evaluation(newData);
        evaluator.crossValidateModel(estimador, newData, 10, new Random(1)); // Random(1): the seed=1 means "no shuffle" :-!
        //  HACER!!!! Imprimir matriz de confusión
        double acc=evaluator.pctCorrect();
        double inc=evaluator.pctIncorrect();
        double kappa=evaluator.kappa();
        double mae=evaluator.meanAbsoluteError();
        double rmse=evaluator.rootMeanSquaredError();
        double rae=evaluator.relativeAbsoluteError();
        double rrse=evaluator.rootRelativeSquaredError();
        double confMatrix[][]= evaluator.confusionMatrix();

        System.out.println("Correctly Classified Instances  " + acc);
        System.out.println("Incorrectly Classified Instances  " + inc);
        System.out.println("Kappa statistic  " + kappa);
        System.out.println("Mean absolute error  " + mae);
        System.out.println("Root mean squared error  " + rmse);
        System.out.println("Relative absolute error  " + rae);
        System.out.println("Root relative squared error  " + rrse);



		/*
		 // 3.2 Alternatively, assess the performance of the classifiera by means of hold-out: leaving the 30% of the data randomly selected out to test the model
		// 3.2.a Get the test set by randomly selecting the the 30% of the instances
		int trainSize = (int) Math.round(newData.numInstances() * 0.7);
		int testSize = newData.numInstances() - trainSize;
		// HACER!!!! Salvar las instancias del test en un fichero
		Instances train = new Instances(newData, 0, trainSize);
		Instances test = new Instances(newData, trainSize, testSize);

		// 3.2.b Train the classifier with the 70\% of the data by means of the Naive Bayes algorithm
		estimador.buildClassifier(train);

		// 3.2.c Let the model predict the class for each instance in the test set
		evaluator.evaluateModel(estimador, test);
		double predictions[] = new double[test.numInstances()];
		for (int i = 0; i < test.numInstances(); i++) {
			predictions[i] = evaluator.evaluateModelOnceAndRecordPrediction((Classifier)estimador, test.instance(i));
		}
		// HACER!!!! Guardar en un fichero de salida la clase estimada por el modelo para cada instancia del test y así después podremos comparar la clase real y la estimada

		// 3.2.d Assess the performance on the test
		//  HACER!!!! Idéntico idéntico idéntico al 3.1: por eso es necesario que sea modular, no vamos a copiar aquí el código de nuevo!
		*/
        ///////////////////////////////////////////////////////
        // Observa: http://weka.wikispaces.com/Use+Weka+in+your+Java+code
        ///////////////////////////////////////////////////////


    }
}