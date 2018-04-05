#include <bits/stdc++.h>
#include "eigen/Eigen/Dense"
#include <string.h>

#define FOR(i, a, b) for(int i=a; i<b; i++)
#define float  double

using namespace Eigen;
using namespace std;

/*
	 data structure to store the data: train data, test data and validation data.
*/
struct data_structure{
	Eigen :: MatrixXd features;
	Eigen :: RowVectorXd target_class;
};

/*
	data structure to store the neural network . Basically its a network with two layers and hence two matrices for weights
*/
struct network{
    Eigen :: MatrixXd i_h;
    Eigen :: MatrixXd h_o;
};

/*
	data structure to store the cost and activations of the hidden layer and the output layer.
*/
struct activations{
    float cost;
    Eigen :: MatrixXd hidden_;
    Eigen :: MatrixXd output_;
};

/*
	data structure to store the gradients of the hidden layer and the output layer.
*/
struct gradients{
    Eigen :: MatrixXd grad_i_h;
    Eigen :: MatrixXd grad_h_o;
};

/*
	Function initialize_network : This function is used to create the neural network and initialize it randomly.

	Input Params:
				@input_layer_size - int : stores the size of the input layer
				@hidden_layer_size - int : stores the size of the hidden layer
				@output_layer_size - int : stores the size of the output layer

	Output Params:
				@network - network : a network (weights matrices)

*/
network initialize_network(int input_layer_size, int hidden_layer_size, int output_layer_size)
{
    network w;
    w.i_h=MatrixXd::Random(input_layer_size,hidden_layer_size)*0.1;
    w.h_o=MatrixXd::Random(hidden_layer_size,output_layer_size)*0.1;
    return w;
}

/*
	Function read_data : This function reads the data from a file and stores it in the data structure.

	Input Params:
				@path - string : path to the file.

	Output Params:
				@data_structure : a data structure that contains the features and the target class.

*/
data_structure read_data(string path)
{
	ifstream readFile;
	readFile.open(path);
    string line, element;

    vector<vector<float> > X;
    vector<float> y;
    int i, prev_comma_index, no_commas;
    float target;
    int line_no=0;
	if(readFile.is_open())
	{
		for(line; getline(readFile, line);)
            {
                vector<float> data_point;
                line_no++;
                no_commas=0;
                prev_comma_index=0;
                for(i=0; i<line.length(); i++)
                    {
                        if(line[i]==',' && no_commas<64)
                        {
                            data_point.push_back(atof(line.substr(prev_comma_index,i-prev_comma_index).c_str()));
                            prev_comma_index=i+1;
                            no_commas++;
                        }
                    }
            y.push_back(atof(line.substr(prev_comma_index,line.length()-prev_comma_index).c_str()));
            X.push_back(data_point);
            }
	}
	Eigen::MatrixXd X_(X.size(),X[0].size());
	for(int i=0;i<X.size();i++)
        for(int j=0;j<X[0].size();j++)
            X_(i,j) = X[i][j];

    Eigen :: RowVectorXd y_(y.size());
    for(int j=0;j<y.size();j++)
            y_(j) = y[j];

    data_structure d;
	d.features = X_;
	d.target_class = y_;
    return d;
}

/*
	Function compute_cost : Computing the cross entropy error

	Input Params:
				@y - Eigen::MatrixXd : data points features;
				@y_true : Eigen::RowVectorXd : data points target classes true values.



	Output Params:
				@cose float : cross entropy error.

*/
float compute_cost(Eigen :: MatrixXd y, Eigen :: RowVectorXd y_true)
{
    float cost=0;
    int n=0;
    FOR(i, 0, y.rows())
        {
            n++;
            cost+=(-log(y(i, y_true(i))));
        }
    cost/=n;
    return cost;
}

/*
	Function sigmoid : get sigmoid of x;

	Input Params:
				@x - float : x

	Output Params:
				@sigmoid - float : 1/(1+e^(-x))
*/
float sigmoid(float x)
{
    return 1.0/(1+exp(-x));
}

/*
	Function Exp : get e^x

	Input Params:
				@x - float : x

	Output Params:
				@sigmoid - float : e^x
*/
double Exp(double x)
{
    return std::exp(x);
}

/*
	Function forward_propagation : This function performs the forward propogation on the neural network.

	Input Params:
				@net - network : the neural network (weights matrices)
				@X - Eigen::MatrixXd : the data that will be used to train the network. Basically a mini batch.


	Output Params:
				@activations - activations : the activations of the hidden and output layer.

*/
activations forward_propagation(network net, Eigen :: MatrixXd X)
{
		int i,j;
    activations act;
    act.hidden_ = (X*net.i_h).unaryExpr(&sigmoid); // op shape= data_size*hidden_layer_size
    act.output_ = (act.hidden_*net.h_o); // op shape= data_size*op_size
		vector<float> denominator_vector ;

		for(i=0;i<X.rows();i++)
		{
				double denominator=0;
				denominator = (act.output_.row(i)).unaryExpr(&Exp).sum();

				denominator_vector.push_back(denominator);
		}

		for(i=0;i<X.rows();i++)
		{
			for(j=0;j<act.output_.cols();j++)
			{
				act.output_(i,j) = Exp(act.output_(i,j))/denominator_vector[i] ;
			}

		}

    return act;
}


/*
	Function one_hot : This function takes a number and converts it into a one hot vector  (for an array of numbers)

	Input Params:
				@vec - Eigen::RowVectorXd : vector of numbers tp be converted to one hot vectors.
				@no_classes - int : Lenght of one hot vector.


	Output Params:
				@matrix - Eigen::MatrixXd  : Matrix of one hot vectors.

*/
Eigen :: MatrixXd one_hot(Eigen :: RowVectorXd vec, int no_classes)
{
    Eigen :: MatrixXd encoded = MatrixXd::Zero(vec.cols(), no_classes);
    for(int i=0; i<vec.cols(); i++)
        encoded(i, vec(i))=1;
    return encoded;
}



/*
	Function gradient_descent : Does the gradient descent by updating the gradients

	Input Params:
				@net - Network & : The neural net (weight matrices)
				@y -  Eigen :: MatrixXd & : data points features
				@current - gradients : current iteration gradient
				@previous - gradients : previous iteration gradient
				@learning_rate float : learning rate
				@momentum float : constant for mommentum

	Output Params:
				@gradients - gradients : current vts which will be used in the next iteration for momentum.

*/
gradients gradient_descent(network &net, Eigen :: MatrixXd &y, gradients current, gradients previous, float learning_rate, float momentum)
{

		gradients vt;
		vt.grad_i_h = momentum*previous.grad_i_h  + learning_rate*current.grad_i_h;
		vt.grad_h_o = momentum*previous.grad_h_o +  learning_rate*current.grad_h_o;

		net.i_h-= vt.grad_i_h;
		net.h_o-=  vt.grad_h_o;

		return vt;

}

/*
	Function backpropagation : This function updates the gradients during the back propogation.

	Input Params:
				@net - network : the neural network (weights matrices)
				@act - activations : activations of hidden and output layer.
				@X - Eigen::MatrixXd : the data that will be used to train the network. Basically a mini batch.
				@y_true - Eigen::RowVectorXd : the vector of true values of the data points.
				@previous - gradients : the gradients in the previous iteration.
				@learning_rate - float : learning rate, default value = 0.01
				@momentum - float : momentum for adam , default value = 0.9

	Output Params:
				@gradients  - gradients : current vts after the update for momentum.

*/
gradients backpropagation(network &net, activations &act, Eigen :: MatrixXd X, Eigen :: RowVectorXd y_true, gradients previous,float learning_rate=0.01, float momentum=0.9)
{
    act.cost = compute_cost(act.output_, y_true);

    gradients current;
		current.grad_h_o = (act.hidden_.transpose())*(act.output_-(one_hot(y_true, 10)));
		current.grad_i_h = (act.output_-(one_hot(y_true, 10)))*net.h_o.transpose();

		Eigen::MatrixXd temp = Eigen::MatrixXd::Ones(act.hidden_.rows(),act.hidden_.cols());
		for (size_t i = 0; i < act.hidden_.rows(); i++) {
			for (size_t j= 0; j < act.hidden_.cols(); j++) {
				temp(i,j) = act.hidden_(i,j)*(1-act.hidden_(i,j)) * current.grad_i_h(i,j);
			}

		}
		current.grad_i_h = X.transpose()*temp;

    return gradient_descent(net, act.output_, current, previous, learning_rate, momentum);
}


/*
	Function predict : used to predict after the training of the neural network

	Input Params:
				@net - network : the neural network (weights matrices)
				@X - Eigen::MatrixXd : the test data that will be used to predict.

	Output Params:
				@predictions - Eigen :: RowVectorXd : predictions .

*/
Eigen :: RowVectorXd predict(network net, Eigen :: MatrixXd X)
{
    activations act;
    Eigen :: RowVectorXd predictions(X.rows());
    act = forward_propagation(net, X);
    MatrixXf::Index maxIndex;
    FOR(i, 0, X.rows())
        {
            act.output_.row(i).maxCoeff(&maxIndex);
            predictions(i)=maxIndex;
        }
    return predictions;
}

/*
	Function accuracy : function to calculate the accuracy of the model.

	Input Params:
				@y - Eigen :: RowVectorXd : the predicted values of target class
				@t - Eigen :: RowVectorXd : the true values of target class


	Output Params:
				@accuracy - float : accuracy in %.

*/
float accuracy(Eigen :: RowVectorXd y, Eigen :: RowVectorXd t)
{
    float count = 0;
    FOR(i, 0, t.cols())
        if(y(i) == t(i))
            count++;
    return 100*(count/y.cols());
}
/*
	Function train : This function trains the neural network.

	Input Params:
				@net - network & : a reference to the initalized network.
				@data - data_structure : train data.
				@val_data - data_structure : validation data.
				@max_iters - int : number of gradient descent steps.


	Output Params:
				none
*/
void train(network &net, data_structure data, data_structure val_data, int max_iters)
{
    int BATCH_SIZE = 100;  //mini batch size.
    int start_row=0;			 // start of batch.
    int end_row= start_row+BATCH_SIZE;   // end of batch.
    int batch_size;
    float prev_accuracy=0;
    float best_accuracy=0;
		float learning_rate = 0.01;
    gradients previous_gradients;
		// initalize gradients to zero.
    previous_gradients.grad_h_o= Eigen :: MatrixXd::Ones(net.h_o.rows(), net.h_o.cols())*0.0;
    previous_gradients.grad_i_h= Eigen :: MatrixXd::Ones(net.i_h.rows(), net.i_h.cols())*0.0;
		int prev = 0;
		int check = 0;
		float curr_accuracy=0;
		int iter;
    for(iter=0; iter< max_iters;iter++)
    {
        batch_size=BATCH_SIZE;
        end_row=start_row+BATCH_SIZE;
        if(end_row>=data.features.rows())   // if the end row exceeds the number of data points then start again.
        {
            start_row=0;
            end_row = start_row+BATCH_SIZE;
        }

        Eigen :: MatrixXd batch_X(batch_size,64);
        Eigen :: RowVectorXd batch_y(batch_size);
        FOR(i,start_row,end_row)     // slicing the batch from the original data set.
        {
            FOR(j, 0, 64)
                batch_X(i-start_row, j) = data.features(i, j);
            batch_y(i-start_row) = data.target_class(i);
        }

        start_row=end_row;
        activations act;
        act = forward_propagation(net, batch_X);

        previous_gradients = backpropagation(net, act, batch_X, batch_y, previous_gradients, learning_rate*(1.0/(1.0+iter/300.0)), 0.9);
				curr_accuracy = accuracy(predict(net, val_data.features), val_data.target_class);
				if(curr_accuracy < prev_accuracy)
            check++;
        else
            {
							check = 0;
							//prev_accuracy = curr_accuracy;
						}
        if(check==100)
            break;
        if(curr_accuracy>best_accuracy)
            best_accuracy=curr_accuracy;
        if(check==0)
					prev_accuracy = curr_accuracy;
    }
    cout<<"Best accuracy = "<<best_accuracy<<"\nHidden layer size = "<<net.i_h.cols()<<"\nTotal number of iterations = "<< iter<<endl;
}


/*
	Function make_confusionmatrix : This function calculates the confusion matrix for the test data

	Input Params:
				@actual_Value - Eigen::RowVectorXd : the vector of true values of the data points.
				@test_predictions - Eigen::RowVectorXd : the vector of predicted values of the data points.


	Output Params:
				@confusion_matrix  -Eigen :: MatrixXd : confusion matrix for the test data
*/
Eigen :: MatrixXd make_confusionmatrix(Eigen :: RowVectorXd actual_Value, Eigen :: RowVectorXd test_predictions)
{
    Eigen::MatrixXd confusion_matrix(10,10);
    confusion_matrix= Eigen :: MatrixXd::Ones(10,10)*0.0;

    for(int i = 0;i<actual_Value.size();i++)
    {
        int act_val = actual_Value(i);
        int pred_val = test_predictions(i);
        //cout<<i<<" "<<act_val << " "<<pred_val<<endl;
        confusion_matrix(act_val,pred_val)+=1;
    }
    return confusion_matrix;
}
/*
	Function evaluate : This function calculates the confusion matrix for the test data

	Input Params:
				@test - data_structure  : Data structure of test data points
				@net - network : the trained neural network


	Output Params:
				None
*/
void evaluate(data_structure test,network &net)
{
   Eigen :: RowVectorXd test_predictions = predict(net, test.features);
   float test_accuracy = accuracy(test_predictions,test.target_class);
   cout<<"Test Accuracy = " << test_accuracy <<endl;
   Eigen::MatrixXd confusion_matrix;
   confusion_matrix = make_confusionmatrix(test_predictions,test.target_class);




    Eigen :: RowVectorXd precision(10);
    Eigen :: RowVectorXd recall(10);
    Eigen :: RowVectorXd f_score(10);


    for(int i = 0;i<10;i++)
    {
        precision(i) = ((float)confusion_matrix(i,i)/(float)(confusion_matrix.col(i).sum()));
        recall(i) = ((float)confusion_matrix(i,i)/(float)(confusion_matrix.row(i).sum()));
        f_score(i) = ((2*precision(i)*recall(i))/(precision(i)+recall(i)));
    }
    cout<<"Confusion Matrix\n" << confusion_matrix<<endl<<endl<<"Precision = "<<precision<<endl<<"Recall = "<<recall<<endl<<"Fscore = "<<f_score<<endl<<endl<<"----------------------------------------------------------------------------------------------------------------------------------------"<<endl;
}
// driver program to run the code.
int main()
{
	data_structure train_data ,test_data, validation_data;
	// reading data.
	test_data=read_data("data/test.txt");
	train_data=read_data("data/train.txt");
	validation_data=read_data("data/validation.txt");

	// creating the network and initializing it.

	for(int i=5; i<=10;i++)
	{
		network network_1;
		network_1 = initialize_network(64, i, 10);

		// training the network with train data.
		train(network_1, train_data, validation_data, 3000);
		evaluate(test_data,network_1);

	}


	return 0;
}
