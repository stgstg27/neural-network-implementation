#include <bits/stdc++.h>
#include "eigen/Eigen/Dense"
#include <string.h>

#define FOR(i, a, b) for(int i=a; i<b; i++)
#define float  double

using namespace Eigen;
using namespace std;

struct data_structure{
	Eigen :: MatrixXd features;
	Eigen :: RowVectorXd target_class;
};

struct network{
    Eigen :: MatrixXd i_h;
    Eigen :: MatrixXd h_o;
};

struct activations{
    float cost;
    Eigen :: MatrixXd hidden_;
    Eigen :: MatrixXd output_;
};

struct gradients{
    Eigen :: MatrixXd grad_i_h;
    Eigen :: MatrixXd grad_h_o;
};


network initialize_network(int input_layer_size, int hidden_layer_size, int output_layer_size)
{
    network w;
    w.i_h=MatrixXd::Random(input_layer_size,hidden_layer_size);
    w.h_o=MatrixXd::Random(hidden_layer_size,output_layer_size);
    return w;
}

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

float sigmoid(float x)
{
    return 1.0/(1+exp(-x));
}

activations forward_propagation(network net, Eigen :: MatrixXd X)
{
    activations act;
    act.hidden_ = (X*net.i_h).unaryExpr(&sigmoid); // op shape= data_size*hidden_layer_size
    act.output_ = (act.hidden_*net.h_o); // op shape= data_size*op_size
    act.output_ -= ;
    // SOFTMAX
    return act;
}

float compute_cost(Eigen :: MatrixXd y, Eigen :: RowVectorXd y_true)
{
    float cost=0;
    int n;
    FOR(i, 0, y.rows())
        {
            n++;
            cost+=(-log(y(i, y_true(0, i))));
        }
    cost/=n;
    return cost;
}

void gradient_descent(network &net, Eigen :: RowVectorXd &y, gradients current, gradients previous, float learning_rate, float momentum)
{
    //current_update;
    vec-=(momentum*prev_update_vector+learning_rate*current_update_vector);
}

void backpropagation(network &net, activations &act, Eigen :: RowVectorXd y_true, gradients previous,float learning_rate=0.01, float momentum=0.9)
{
    act.cost = compute_cost(act.output_, y_true);

    gradients current;
    current.grad_h_o = (act.output_-y_true).array()*act.hidden_.array();
    cout<<"grads: "<<current.grad_h_o.rows()<<' '<<current.grad_h_o.cols()<<endl;
    // current.grad_h_o
    // current.grad_h_o = ;

    //gradient_descent(net, act.output_, current, previous, learning_rate, momentum);
    return current;
}

void train(network &net, data_structure data, data_structure val_data, int max_iters)
{
    int BATCH_SIZE = 100;
    int start_row=0;
    int end_row= start_row+BATCH_SIZE;
    int batch_size;
    float prev_accuracy=101;
    FOR(iter, 0, max_iters)
    {
        batch_size=BATCH_SIZE;
        end_row=start_row+BATCH_SIZE;
        if(end_row>=data.features.rows())
        {
            start_row=0;
            end_row = start_row+BATCH_SIZE;
        }
        Eigen :: MatrixXd batch_X(batch_size,64);
        Eigen :: RowVectorXd batch_y(batch_size);
        FOR(i,start_row,end_row)
        {
            FOR(j, 0, 64)
                batch_X(i-start_row, j) = data.features(i, j);
            batch_y(i-start_row) = data.target_class(i);
        }
        start_row=end_row;
        activations act;
        act = forward_propagation(net, batch_X);
        previous_gradients = backpropagation(net, act, batch_y, previous_gradients, 0.01, 0.9);
        cout<<"step: "<<iter+1<<" cost: "<<act.cost<<endl;
        if(accuracy(predict(net, batch_X), batch_y)<prev_accuracy)
            break;
    }
    cout<<"best accuracy: "<<prev_accuracy<<endl;
}

Eigen :: RowVectorXd predict(network net, data_structure X)
{
    activations a;
    Eigen :: RowVectorXd ans(X.rows());
    a = forward_propagation(net, X);
    FOR(i, 0, X.rows())
        ans(i) = a.output_.row(i).maxCoeff();
    return ans;
}

float accuracy(Eigen :: RowVectorXd y, Eigen :: RowVectorXd y_)
{
    float count = 0;
    FOR(i, 0, y_.cols())
        if(y(i) == y_(i))
            count++;
    return 100*(count/y.cols());
}

int main()
{
	data_structure train_data ,test_data, validation_data;
	train_data=read_data("train.txt");
	test_data=read_data("test.txt");
	validation_data=read_data("validation.txt");

	network network_1;
	network_1 = initialize_network(64, 10, 10);
	train(network_1, train_data, validation_data, 3000);

	//Eigen :: RowVectorXd predicted=predict(network_1, validation_data.features);
	//cout<<accuracy(predicted, validation_data.target_class)<<endl;
	return 0;
}
