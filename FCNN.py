import tensorflow as tf
import tensorflow_datasets as tfds 

# Loading data
(ds_train, ds_test), ds_info = tfds.load(
    'mnist', #data from the database 'mnist'  
    split=['train', 'test'], #multiple datasets are returned separately : separation into training and test base 
    shuffle_files=True, #mixing of data because of voluminous data
    as_supervised=True, #returns a tuple instead of a dictionary
    with_info=True, #returns the information about a dataset (name, version, and features)
)


# Normalizing function
def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label #normalizes images: uint8 -> float32

# Training pipeline
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE) #parallelization enabled to speed up processing 
ds_train = ds_train.cache() #dataset caching (data loaded into memory after the first training epoch in order to speed up other epochs)
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) #grouping of elements in order to obtain unique batches at each epoch
ds_train = ds_train.batch(128) #groups the elements of the dataset into mini-batches of size 128
ds_train = ds_train.prefetch(tf.data.AUTOTUNE) #preread to improve performances 

# Test pipeline 
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE) #parallelization enabled to speed up processing
ds_test = ds_test.batch(128) #groups the elements of the dataset into mini-batches of size 128
ds_test = ds_test.cache() #dataset caching (done after batch processing because batches may be the same from one epoch to another) 
ds_test = ds_test.prefetch(tf.data.AUTOTUNE) #preread to improve performances 


def train_model(nb_epochs, learning_rate, nb_neurals, activation_function):

    # Creating the model 
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #flattens the input (converting 2D images into a 1D vector of 784 elements)
    tf.keras.layers.Dense(nb_neurals, activation_function), #definition of a dense layer (with neurals and activation function)
    tf.keras.layers.Dense(10) #definition of a dense layer (10 outputs for 10 labels (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) 
    ])

    # Configure the model for training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate), #using the Adam optimizer with a certain learning rate 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#sets the loss function to Sparse Categorical Crossentropy, the model's output is not normalized
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()], #calculates the accuracy between the predicted and true labels
    )

    # Training the model 
    modele = model.fit(
        ds_train, #train dataset (model will be trained on it)
        epochs=nb_epochs, #epochs
        validation_data=ds_test,#test dataset(comparison data to evaluate model performance)
    )

    # Extracting relevant metrics from training the model 
    loss = modele.history['loss'][-1]
    sparse_categorical_accuracy = modele.history['sparse_categorical_accuracy'][-1] 
    val_loss = modele.history['val_loss'][-1]
    val_sparse_categorical_accuracy = modele.history['val_sparse_categorical_accuracy'][-1]

    return loss, sparse_categorical_accuracy, val_loss, val_sparse_categorical_accuracy  


# Parameters to change to improve performances (parameters can be added to find the best ones)
nb_epochs_values = [5, 10, 20, 50] 
learning_rate_values = [0.001, 0.01, 0.1, 0.3, 0.5]
nb_neurals_values = [270, 350, 397, 420, 500] #(28*28+10)/2 = 397 (mean of inputs and outputs)
activation_functions = ['relu', 'sigmoid', 'tanh',]
#relu function (Rectified Linear Unit) : replaces all negative values ​​with zero and leaves positive values ​​unchanged
#sigmoid function : compresses values ​​between 0 and 1
#tannh function (Hyperbolic tangent) : compresses values ​​between -1 and 1

# Stock the results and the evaluation metrics  
results = [] 
losses = [] 
sparse_categorical_accuracy_values =[]
val_losses = []
val_sparse_categorical_accuracy_values = []

# Loop to find the best parameter combinations 
for nb_epochs in nb_epochs_values:
    for learning_rate in learning_rate_values:
        for nb_neurals in nb_neurals_values:
            for activation_function in activation_functions: 
                loss, sparse_categorical_accuracy, val_loss, val_sparse_categorical_accuracy = train_model(nb_epochs, learning_rate, nb_neurals, activation_function) 
                
                # Stockage of the values 
                losses.append(loss)
                sparse_categorical_accuracy_values.append(sparse_categorical_accuracy)
                val_losses.append(val_loss)
                val_sparse_categorical_accuracy_values.append(val_sparse_categorical_accuracy) 

                # Stock results for each combination
                results.append({
                    'nb_epochs': nb_epochs,
                    'learning_rate': learning_rate,
                    'nb_neurals': nb_neurals,
                    'activation_function': activation_function,
                    'loss': loss,
                    'sparse_categorical_accuracy': sparse_categorical_accuracy,
                    'val_loss': val_loss,
                    'val_sparse_categorical_accuracy': val_sparse_categorical_accuracy
                })


# Find the best result based on various criteria
best_result_min_loss = min(results, key=lambda x: x['loss']) #loss : perte (différence entre la valeur prédite et celle attendue (base d'entrainement)) - doit diminuer pour améliorer les performances 
best_result_max_accuracy = max(results, key=lambda x: x['sparse_categorical_accuracy']) #sparse_categorical_accuracy : précision catégorique éparse (comparaison avec la base d'entrainement) - doit augmenter pour améliorer les performances 
best_result_min_val_loss = min(results, key=lambda x: x['val_loss']) #val_loss : perte (différence entre la valeur prédite et celle attendue (base de test)) - doit diminuer pour améliorer les performances
best_result_max_val_accuracy = max(results, key=lambda x: x['val_sparse_categorical_accuracy']) #val_sparse_categorical_accuracy: précision catégorique éparse (comparaison avec la base de test) - doit augmenter pour améliorer les performances

#On aurait pu créer une fonction qui, pour un seuil donné, considère que les métriques de l'apprentissage ne sont pas cohérentes et donc le supprime et on prend le meilleure résultats parmis les données restantes: 

# Display the best combination with its parameters 
print(f"Perte minimale sur les données d'apprentissage : {best_result_min_loss}")
print(f"Combinaison de paramètres - nombre d'époques: {best_result_min_loss['nb_epochs']}, learning rate: {best_result_min_loss['learning_rate']}, nombre de couches de neurones : {best_result_min_loss['nb_neurals']}, fonction d'activation : {best_result_min_loss['activation_function']}")
print("\n")

print(f"Précision catégorique éparse maximale sur les données d'apprentissage : {best_result_max_accuracy}")
print(f"Combinaison de paramètres - nombre d'époques: {best_result_max_accuracy['nb_epochs']}, learning rate: {best_result_max_accuracy['learning_rate']}, nombre de couches de neurones: {best_result_max_accuracy['nb_neurals']}, fonction d'activation: {best_result_max_accuracy['activation_function']}")
print("\n")

#Observation : on affiche les résultats d'évalutation pour les données d'apprentissage afin de vérifier qu'il n'y a pas d'incohérence dans les résultats. 

print(f"Perte minimale sur les données de test : {best_result_min_val_loss}")
print(f"Combinaison de paramètres - nombre d'époques: {best_result_min_val_loss['nb_epochs']}, learning rate: {best_result_min_val_loss['learning_rate']}, nombre de couches de neurones: {best_result_min_val_loss['nb_neurals']}, fonction d'activation: {best_result_min_val_loss['activation_function']}")
print("\n")

print(f"Précision catégorique éparse maximale sur les données de test : {best_result_max_val_accuracy}")
print(f"Combinaison de paramètres - nombre d'époques: {best_result_max_val_accuracy['nb_epochs']}, learning rate: {best_result_max_val_accuracy['learning_rate']}, nombre de couches de neurones: {best_result_max_val_accuracy['nb_neurals']}, fonction d'activation: {best_result_max_val_accuracy['activation_function']}")
print("\n")  





