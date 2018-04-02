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

float softmax(float x)
{
    return 1.0/(1+exp(x));
}



activations forward_propagation(network net, Eigen :: MatrixXd X)
{
    activations act;
    act.hidden_ = X*net.i_h; // op shape= data_size*hidden_layer_size
    act.output_ = (act.hidden_*net.h_o).unaryExpr(&softmax); // op shape= data_size*op_size
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

void backpropagation(network &net, activations &act, Eigen :: RowVectorXd y_true)
{
    float cost=0;
    act.cost = compute_cost(act.output_, y_true);
}

void train(network &net, data_structure data,int max_iters)
{
    int BATCH_SIZE = 100;
    int start_row=0;
    int end_row= start_row+BATCH_SIZE;
    int batch_size;
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
        end_row=start_row+BATCH_SIZE;
        activations act;
        act = forward_propagation(net, batch_X);
        backpropagation(net, act, batch_y);
        cout<<"step: "<<iter+1<<" cost: "<<act.cost<<endl;
    }
}


int main()
{
	data_structure train_data ,test_data, validation_data;
	train_data=read_data("train.txt");
	test_data=read_data("test.txt");
	validation_data=read_data("validation.txt");

	network network_1;
	network_1 = initialize_network(64, 10, 10);
	train(network_1, train_data, 3000);
	//predict(network_1, validation_data);

	return 0;
}
