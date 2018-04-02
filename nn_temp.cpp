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
    Eigen :: MatrixXd i_h;
    Eigen :: MatrixXd h_o;
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
                //cout<<"line: "<<line_no<<endl;
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
	//cout<<X_.rows()<<' '<<X_.cols()<<endl;
	d.target_class = y_;
	//cout<<y_.cols()<<endl;
    return d;
}

activations forward_propagation(network net, data_structure data)
{

}

void backpropagation(network &net, activations act)
{

}

void train(network &net, data_structure data,int max_iters)
{
    FOR(iter, 0, max_iters)
    {
        act=forward_propagation(net, data)
        backpropagation(net, act)
        cout<<"step: "<<iter<<" cost: "<<act.cost<<endl;
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
	train(network_1, 3000);
	predict(network_1, validation_data);

	return 0;
}
