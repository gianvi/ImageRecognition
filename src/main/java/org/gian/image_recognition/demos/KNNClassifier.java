package org.gian.image_recognition.demos;

import org.gian.image_recognition.core.data.FVData;
import org.gian.image_recognition.core.extractors.CFIIFExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.knn.DoubleNearestNeighbours;
import org.openimaj.knn.approximate.DoubleNearestNeighboursKDTree;
import org.openimaj.util.pair.Pair;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * A class and tester for testing KNN classification of Images
 * which are built randomly from the {@link CFIIFExtractor}.
 * <p>
 * In this implementation, the CFIIF features vector of the images are
 * used as the vector for classification which will achieve somewhere in the
 * region of 60% correct classification for the characters 0-9 and somewhat
 * worse for larger character ranges.
 * <p>
 * Override the {@link #getImageVector(FImage)} method to try different
 * features. This class provides the training, classification and testing.
 *
 * @author Fase
 * @created 22 Feb 2016
 */
public class KNNClassifier {
    /**
     * This class receives each image as an FVData
     * When it receives an image, it creates(get) the feature vector by calling
     * {@link KNNClassifier#getImageVector(FImage)} and stores that
     * into a double array that can be used as input to the nearest neighbour
     * classifier for training.
     * The output feature matrix can be retrieved using getVectors().
     *
     * @author Fase
     * @created 22 Feb 2016
     */
    private ImageTrainer imgTrainer;
    private ImageTester imgTester;

    protected class ImageTrainer {
        /**
         * map for indexVector - path, label
         */
        private Map<Integer, ArrayList<String>> imgMap;

        /**
         * The output matrix that can be used as input to the NN classifier
         */
        private double[][] vector = null;

        /**
         * The current index being processed
         */
        private int index;

        /**
         * Constructor that takes training instances and initialize the output vector matrix
         *
         * @param trainImages List of training images with feature vector
         */
        public ImageTrainer(List<FVData> trainImages) {
            this.index = 0;
            vector = new double[trainImages.size()][];
            imgMap = new HashMap<Integer, ArrayList<String>>();

            for (FVData img : trainImages) {
                //riempi la matrice
                //puo esser utile fare una mappa path - label, featVector

                vector[index] = img.getFeatureVector().asDoubleVector();

                ArrayList<String> ls = new ArrayList<String>();
                ls.add(img.getPath());
                ls.add(img.getType());
                imgMap.put(index, ls);

                index++;
            }
            index--;
        }


        /**
         * Retrieve the set of training data as a double array.
         *
         * @return the training data
         */
        public double[][] getVectors() {
            return vector;
        }
    }

    protected class ImageTester {

        /**
         * map for indexVector - path, label
         */
        private Map<Integer, ArrayList<String>> imgMap;

        /**
         * The output matrix that can be used as input to the NN classifier
         */
        private double[][] vector = null;

        /**
         * The current index being processed
         */
        private int index;

        /**
         * Constructor that takes training instances and initialize the output vector matrix
         *
         * @param testImages List of test images with feature vector
         */
        public ImageTester(List<FVData> testImages) {
            this.index = 0;
            vector = new double[testImages.size()][];
            imgMap = new HashMap<Integer, ArrayList<String>>();

            for (FVData img : testImages) {
                //riempi la matrice
                //puo esser utile fare una mappa path - label, featVector

                vector[index] = img.getFeatureVector().asDoubleVector();

                ArrayList<String> ls = new ArrayList<String>();
                ls.add(img.getPath());
                ls.add(img.getType());

                imgMap.put(index, ls);

                index++;
            }
            index--;
        }


        /**
         * Retrieve the set of training data as a double array.
         *
         * @return the training data
         */
        public double[][] getVectors() {
            return vector;
        }

        public double[] getVector(int index) {
            return vector[index];
        }
    }


    private int NTREES = 768;
    private int NCHECKS = 8;

    /**
     * The nearest neighbour classifier
     */
    private DoubleNearestNeighbours nn = null;
    

    /**
     * Default constructor: split datesaet into train/test sets
     */
    public KNNClassifier(List<FVData> corpus) {
        //train 80%
        int trainSize = Math.round(corpus.size() * 60 / 100);

        //test 20%
        int testSize = corpus.size() - trainSize;

        List<FVData> shuffled = pickNRandom(corpus);

        this.imgTrainer = new ImageTrainer(shuffled.subList(0, trainSize - 1));

        this.imgTester = new ImageTester(shuffled.subList(trainSize, corpus.size() - 1));

        System.out.println("\n 1) SPLIT DATASET[" + corpus.size() + "]: \n  trainSET[" + trainSize + "] \n testSET[" + testSize + "]\n");

    }


    private static List<FVData> pickNRandom(List<FVData> lst) {
        List<FVData> copy = new LinkedList<FVData>(lst);
        Collections.shuffle(copy);
        return copy;
    }


    /**
     * Get the feature vector for a single image. Can be overridden to
     * try different feature vectors for classification.
     *
     * @param img The character image.
     * @return The feature vector for this image.
     */
    public double[] getImageVector(FImage img) {
        // Resize the image (stretch) to a standard shape
        //FImage ff = ResizeProcessor.resample( img, resize, resize );
        //return ff.getDoublePixelVector();
        return null;
    }

    /**
     * Train the classifier by loading training examples
     * feature vectors of various labels
     * (using the {@link KNNClassifier.ImageTrainer})
     * Using the features (already) extracted from those images, train a nearest
     * neighbour classifier.
     */
    public void train() {
        nn = new DoubleNearestNeighboursKDTree(this.imgTrainer.getVectors(), NTREES, NCHECKS);
        System.out.println("\n 2) TRAINING con " + nn.size() + " train images [" + nn.numDimensions() + "]");
    }

    /**
     * Classify the given image with the nearest neighbour classifier.
     *
     * @param img the feature vector image to classify
     * @return The classified image.
     */
    public ArrayList<String> classify(double[] img) {
        // Create the input vector
        double[][] v = new double[1][];
        v[0] = img;

        // Setup the output variables
        int k = 1;
        int[][] indices = new int[1][k];
        double[][] dists = new double[1][k];

        // Do the search
        nn.searchKNN(v, k, indices, dists);

        // Work out what image the best index represents
        System.out.println("Best index: " + indices[0][0]);
        
        return imgTrainer.imgMap.get(indices[0][0]);
    }

    /**
     * Run a bunch of tests to determine how good the classifier is. It does
     * this by creating a load of random examples of random characters using
     * the {@link KNNClassifier.ImageTester} and classifies them.
     */
    public void test() {
        System.out.println("\n 3) TESTING KDTree with " + imgTester.index + " test images");

        // First in the pair is the incorrectly classified count,
        // the second is the correctly classified count.
        final Pair<Integer> results = new Pair<Integer>(0, 0);

        // Loop through all the tests
        for (int j = 0; j < imgTester.index; j++) {

            double[] test = imgTester.getVector(j);
            String label = imgTester.imgMap.get(j).get(1).toUpperCase();
            String id = imgTester.imgMap.get(j).get(0);


            System.out.println("Classifying " + id.substring(id.length() - 15) + " ( " + label + " )");
            ArrayList<String> bestNeighbour = classify(test);
            String predLabel = bestNeighbour.get(1).toUpperCase();
            String predId = bestNeighbour.get(0);
            System.out.println("Predicted " + predId.substring(id.length() - 15) + " ( " + predLabel + " )");

            try {
                MBFImage imageToClassify = ImageUtilities.readMBF(new File(id));
                MBFImage imagePredicted = ImageUtilities.readMBF(new File(predId));

                DisplayUtilities.display(imageToClassify, "Image to classify");

                DisplayUtilities.display(imagePredicted, "Nearest image (predicted)");

            } catch (IOException e) {
                e.printStackTrace();
            }

            // Update the test count
            if (predLabel.equalsIgnoreCase(label)) {
                results.setSecondObject(results.firstObject().intValue() + 1);
            } else {
                results.setFirstObject(results.secondObject().intValue() + 1);
            }
        }


        // Show the test results
        int nWrong = results.firstObject();
        int nCorrect = results.secondObject();

        System.out.println("===========================================");
        System.out.println("           T E S T   R E S U L T S         ");
        System.out.println("===========================================");
        System.out.println("Number of runs: " + imgTester.index);
        System.out.println("Correct: " + nCorrect + " (" + (100 * nCorrect / imgTester.index) + "%)");
        System.out.println("Wrong:   " + nWrong + " (" + (100 * nWrong / imgTester.index) + "%)");
        System.out.println("===========================================");

    }


}