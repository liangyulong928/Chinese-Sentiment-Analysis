echo "==========================================================\n"
echo "start project\n"
echo "==========================================================\n"
echo "start emotional dictionary classifier\n"
python EmotionalDictionaryClassifier.py
echo "==========================================================\n"
echo "finish emotional dictionary classifier\n"
echo "==========================================================\n"
echo "start bayesian classifier\n"
python Bayesian.py
echo "==========================================================\n"
echo "finish bayesian classifier\n"
echo "==========================================================\n"
echo "start Convolutional Neural Network\n"
python CNNClassifier.py
echo "==========================================================\n"
echo "finish Convolutional Neural Network\n"
echo "==========================================================\n"