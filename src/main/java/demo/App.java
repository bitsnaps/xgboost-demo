package demo;

import java.io.*;
import java.util.*;
import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;

public class App {

    public static InputStream getResourceAsStream(String name) {
        InputStream result = App.class.getResourceAsStream(name);
        if (result == null) {
            throw new MissingResourceException(
                    String.format("[xgboost-predictor] not found: class = %s, name = %s",
                            App.class.getCanonicalName(), name),
                    App.class.getCanonicalName(), name);
        }

        return result;
    }
    
    public static void main(String[] args) throws java.io.IOException {
        // If you want to use faster exp() calculation, uncomment the line below
        // ObjFunction.useFastMathExp(true);

        // Load model and create Predictor
        Predictor predictor = new Predictor(new java.io.FileInputStream("src/main/resources/model/gbtree/v40/binary-logistic.model"));


        // Create feature vector from dense representation by array
        double[] denseArray = {0, 0, 32, 0, 0, 16, -8, 0, 0, 0};
        FVec fVecDense = FVec.Transformer.fromArray(
                denseArray,
                true /* treat zero element as N/A */);

        // Create feature vector from sparse representation by map
        FVec fVecSparse = FVec.Transformer.fromMap(
                new java.util.HashMap<Integer, Double>() {{
                    put(2, 32.);
                    put(5, 16.);
                    put(6, -8.);
                }});

        // Predict probability or classification
        double[] prediction = predictor.predict(fVecDense);

        // prediction[0] has
        //    - probability ("binary:logistic")
        //    - class label ("multi:softmax")

        // Predict leaf index of each tree
        int[] leafIndexes = predictor.predictLeaf(fVecDense);
        
        System.out.println("Length of leafIndexes: " + leafIndexes.length);

        // leafIndexes[i] has a leaf index of i-th tree        
        System.out.println("Done.");
    }
}
