package org.gian.image_recognition.demos;

import org.gian.image_recognition.core.data.*;
import org.gian.image_recognition.utils.BasicGroupedDataset;
import org.gian.image_recognition.utils.ImageAnalysis;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.io.File;
import java.io.IOException;
import java.util.*;


public class FeatureExtractionDemo {

    //TODO variabili replicate nel CFIIFextractor: occhio!
    public static int NUM_TRAINING_IMAGES_PER_CLASS = 15;
    public static int NUM_TEST_IMAGES_PER_CLASS = 15;

    //indica la percentuale sul num complessivo dei kps
    public static double NUM_PERCENTAGE_CLUSTER = 1;

    public static String SUBPATH_TO_SAVE = "TR"+NUM_TRAINING_IMAGES_PER_CLASS+"_CL"+NUM_PERCENTAGE_CLUSTER+"/";

    /**
     * First stage of the image recognition process. Writes each image path and its
     * corresponding classification type to a file.
     * <p>
     * Input:
     * - The root directory of the training data
     * Ouput:
     * - file containing path/type pairs for each image:
     * img1Path car
     * img2Path car
     * img3Path bicycle
     * img4Path bicycle
     * img5Path motorbike
     * img6Path motorbike
     * ....
     */
    public static void loader() throws IOException {
        File imageRoot = new File("imagecl/train/");

        //Prepare training data
        List<TypeData> trainingData = new ArrayList<TypeData>();
        for (File folder : Arrays.asList(imageRoot.listFiles())) {
            if (folder.isDirectory()) {
                Iterable<File> files = Arrays.asList(folder.listFiles()).subList(0, NUM_TRAINING_IMAGES_PER_CLASS);
                for (File file : files) {
                    trainingData.add(new LocalTypeData(file.getAbsolutePath(), folder.getName()));
                }
            }
        }

        //Write classification data to file
        LocalImageData.writeList("tmp/"+SUBPATH_TO_SAVE+"1traindata.txt", trainingData);
    }

    /**
     * Second stage of the image recognition process. Generates keypoints for each image.
     * <p>
     * CAN BE PERFORMED WITH A MAPREDUCE TASK!
     * <p>
     * Input:
     * - file containing path/type pairs for each image
     * Ouput:
     * - file containing path/type/keypoint pairs for each image
     * img1Path car kp1 kp2 ...
     * img2Path car kp1 kp2 ...
     * img3Path bicycle kp1 kp2 ...
     * img4Path bicycle kp1 kp2 ...
     * img5Path motorbike kp1 kp2 ...
     * img6Path motorbike kp1 kp2 ...
     * ....
     */
    public static void kpExtractor() throws IOException {
        //Read training data
        List<TypeData> trainingData = LocalTypeData.readList("tmp/"+SUBPATH_TO_SAVE+"1traindata.txt");

        //Generate keypoints
        List<KPData> keypoints = new ArrayList<KPData>();
        for (TypeData type : trainingData) {
            System.out.println("Generating training image keypoints");

            AbstractDenseSIFT<FImage> sift = ImageAnalysis.getSift();
            sift.analyseImage(type.getImage());
            keypoints.add(new LocalKPData(type.getPath(), type.getType(), sift.getByteKeypoints(0.005f)));
        }

        //Write keypoints to file
        LocalImageData.writeList("tmp/"+SUBPATH_TO_SAVE+"2keypoints.txt", keypoints);
    }

    /**
     * Third stage of the image recognition process. Trains the quantizer based on the given keypoints.
     * <p>
     * CANNOT BE DONE AS A MAPREDUCE TASK! Must be done on a single machine
     * <p>
     * Input:
     * - file containing path/type/keypoint pairs for each image
     * Ouput:
     * - file containing the training data centroids.
     */
    public static void trainer() throws IOException {
        //Read keypoints
        List<KPData> keypoints = LocalKPData.readList("tmp/"+SUBPATH_TO_SAVE+"2keypoints.txt");

        //Train quantizer
        System.out.println("Training quantizer (this may take several minutes...)");

        ByteCentroidsResult centroids = ImageAnalysis.trainQuantiser(keypoints, NUM_PERCENTAGE_CLUSTER);

        LocalCentroidsData data = new LocalCentroidsData(centroids);
        data.write("tmp/"+SUBPATH_TO_SAVE+"3centroids.ser");
    }

    /**
     * Fourth stage of the image recognition process. Trains the quantizer based on the given keypoints.
     * <p>
     * CAN BE PERFORMED WITH A MAPREDUCE TASK!
     * <p>
     * Input:
     * - file containing path/type/keypoint pairs for each image
     * - file containing the training data centroids.
     * Ouput:
     * - file containing path/type/featurepoint pairs for each image
     * img1Path car fp1 fp1 ...
     * img2Path car fp1 fp2 ...
     * img3Path bicycle fp1 fp2 ...
     * img4Path bicycle fp1 fp2 ...
     * img5Path motorbike fp1 fp2 ...
     * img6Path motorbike fp1 fp2 ...
     * ....
     */
    public static void fvExtractor() throws IOException {
        //Read keypoints
        List<KPData> keypoints = LocalKPData.readList("tmp/"+SUBPATH_TO_SAVE+"2keypoints.txt");

        //Generate feature vectors
        List<FVData> features = new ArrayList<FVData>();
        for (KPData kpdata : keypoints) {
            System.out.println("Generating training image features");

            //Read assigner from file
            CentroidsData centroidsData = LocalCentroidsData.read("tmp/"+SUBPATH_TO_SAVE+"3centroids.ser");
            HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

            DoubleFV fv = ImageAnalysis.extractFeatures(assigner, kpdata);
            features.add(new LocalFVData(kpdata.getPath(), kpdata.getType(), fv));
        }

        //Write features to file
        LocalImageData.writeList("tmp/"+SUBPATH_TO_SAVE+"4features.txt", features);
    }

    public static void myFvExtractor() throws IOException {
        //Read images
        List<KPData> imms = LocalKPData.readList("tmp/"+SUBPATH_TO_SAVE+"2keypoints.txt");

        //Generate feature vectors
        System.out.println("Nel corpus ci sono: " + imms.size() + " immagini...");

        //training del modello con tutte le immagini
        System.out.println("\n\n\nGenerating training (CF+IIF) image features");

        //Read assigner from file
        CentroidsData centroidsData = LocalCentroidsData.read("tmp/"+SUBPATH_TO_SAVE+"3centroids.ser");
        HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();


        //Generate feature vectors
        ImageAnalysis.extractMyFeatures(assigner, imms);
        //System.out.println("..ed ora che lo hai estratto salva il featVector:  \n  length: "+fv.length()+" \n  vectorLength: "+fv.getVector().length+"  \n --- Values: 3DECIMAL");


        //LocalImageData.writeList("tmp/4myFeatures.txt", features);
    }

    /**
     * Fifth stage of the image recognition process. Reads feature vectors and
     * uses them to train the knn classificator.
     * <p>
     * In this demo, train and test images are splitted in the knn impl.
     * <p>
     * the CFIIF feature vector are used to the classification task:
     * <p>
     * - knn distance metrics: COSINE
     * - knn distance weighting: TODO
     * - set K: TODO
     * <p>
     * Input:
     * - file containing path/type/featurepoint pairs for each image
     * Output:
     * - The classification of the given test image
     */
    public static void knnClassifier() throws IOException {

        //Read features
        List<FVData> features = LocalFVData.readList("tmp/4myFeatures.txt");

        System.out.println("Create the memory-based KNN classifier, with all images (will be splitted into train/test)");
        KNNClassifier classifier = new KNNClassifier(features);

        classifier.train();
        classifier.test();
    }

    /**
     * Fifth stage of the image recognition process. Reads the computed centroids and feature vectors and
     * uses them to classify images.
     * <p>
     * In this demo, classified test images are used to determine the effectiveness of the image recognizer.
     * <p>
     * Input:
     * - file containing path/type/featurepoint pairs for each image
     * - file containing the training data centroids.
     * Output:
     * - The classification of the given test image
     */
    public static void recognizer() throws IOException {
        //Read features
        List<FVData> features = LocalFVData.readList("tmp/4features.txt");

        CentroidsData centroidsData = LocalCentroidsData.read("tmp/3centroids.ser");
        HardAssigner<byte[], float[], IntFloatPair> assigner = centroidsData.getCentroids().defaultHardAssigner();

        //Train annotator
        System.out.println("Training annotator");
        LiblinearAnnotator<FVData, String> ann = ImageAnalysis.trainAnnotator(features);

        Set<String> annotazioni = ann.getAnnotations();

        for (String s : annotazioni) {
            System.out.println(s);
        }


        //Prepare test data
        BasicGroupedDataset<FVData> testDataset = ImageAnalysis.prepareTestData("imagecl/train", assigner, NUM_TEST_IMAGES_PER_CLASS, NUM_TRAINING_IMAGES_PER_CLASS);


        //Test accuracy rate of annotator
        ClassificationEvaluator<CMResult<String>, String, FVData> eval =
                new ClassificationEvaluator<CMResult<String>, String, FVData>(
                        ann, testDataset, new CMAnalyser<FVData, String>(CMAnalyser.Strategy.SINGLE));
        Map<FVData, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);
        System.out.println(result);
    }

    /**
     * Demonstrates an outline of each step of the image recognition process using the local filesystem.
     * <p>
     * The image recognition process has been broken up into 5 distinct stages. See the documentation of
     * each stage below for more info!
     */
    public static void main(String[] args) throws IOException {
        //Basic system verification
        if (System.getProperty("user.dir").contains(" "))
            throw new RuntimeException("Your project folder path cannot contain spaces!");

        final String dir = System.getProperty("user.dir");
        System.out.println("current dir = " + dir);


        if (!new File("imagecl/").exists())
            throw new RuntimeException("Please download and configure the demo images before proceeding");

        //Run all demo stages


        loader();
        kpExtractor();
        trainer();
        fvExtractor();
        myFvExtractor();

        //contain 2 step actually: train/test (activate just test!)
        //TODO save the model to avoid re.train
        knnClassifier();

        //recognizer();
    }
}
