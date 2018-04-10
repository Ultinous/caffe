#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/ultinous/projective_transformer_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void copy_values(const int nthreads, int size_src, int k, 
	const Dtype* src, int size_dst, int i, Dtype* dst) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		dst[index * size_dst + i] = src[index * size_src + k];
	}
}

template <typename Dtype>
__global__ void ProjectiveTransformerForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* U, Dtype* V, bool grid) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;
		const int row_idx = output_W_ * s + t;

    // homogen coordinates
      const Dtype hom_x = coordinates[row_idx * 3];
      const Dtype hom_y = coordinates[row_idx * 3 + 1];
      const Dtype hom_h = coordinates[row_idx * 3 + 2];

      const Dtype px = hom_x / hom_h;  // NOTE: check for div by zero
      const Dtype py = hom_y / hom_h;

	  	const int V_offset = index;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

	  	int m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}
	  	
	  	if( grid && ( ((int(round(px*64))%7)==0) || ((int(round(py*64))%7)==0) ) )
		  V[V_offset] = (Dtype)0.;
  }
}

template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {



	string prefix = "ProjectiveTransformerLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid.gpu_data();

	Dtype* input_grid_data = input_grid.mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();


	caffe_gpu_set(input_grid.count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);





// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 3, 3, (Dtype)1.,
				output_grid_data, theta + 9 * i, (Dtype)0.,
				input_grid_data + (output_H_ * output_W_ * 3) * i);
	}

	const int nthreads = N * C * output_H_ * output_W_;

	ProjectiveTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS,0,Caffe::cuda_stream()>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V, this->layer_param_.proj_trans_param().grid());
}




template <typename Dtype>
__global__ void ProjectiveTransformerBackwardGPU_dTheta(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* dV_array, const Dtype* U_array,  
		Dtype* dTheta_tmp_diff, const Dtype* theta) {
	
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

    Dtype t11=theta[9*i], t12=theta[9*i+1], t13=theta[9*i+2];
    Dtype t21=theta[9*i+3], t22=theta[9*i+4], t23=theta[9*i+5];
    Dtype t31=theta[9*i+6], t32=theta[9*i+7], t33=theta[9*i+8];

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 3) * i;

		const int row_idx = output_W_ * s + t;

    const Dtype hom_x = coordinates[row_idx * 3];
    const Dtype hom_y = coordinates[row_idx * 3 + 1];
    const Dtype hom_h = coordinates[row_idx * 3 + 2];

    const Dtype px = hom_x / hom_h;  // NOTE: check for div by zero
    const Dtype py = hom_y / hom_h;
		
		Dtype delta_dpx = (Dtype)0.;
		Dtype delta_dpy = (Dtype)0.;

		const Dtype x = (px + 1) / 2 * H;
		const Dtype y = (py + 1) / 2 * W;
		const int dV_offset = index;
		const Dtype dV = dV_array[dV_offset];

		int m, n; 
		const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

		// left-bottom neighbor
		m = floor(x); n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}
		
		// left-top neighbor
		m = floor(x); n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
		}

		// right-bottom neighbor
		m = floor(x) + 1; n = floor(y); 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
			delta_dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		// right-top neighbor
		m = floor(x) + 1; n = floor(y) + 1; 
		if(m >= 0 && m < H && n >= 0 && n < W) {
			delta_dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
			delta_dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
		}
		
		int idx = j * (output_H_ * output_W_) + s * output_W_ + t;

    Dtype outX = (s * 1.0 / output_H_ * 2 - 1);
    Dtype outY = (t * 1.0 / output_W_ * 2 - 1);

    Dtype nomX = t11*outX + t12*outY + t13;
    Dtype nomY = t21*outX + t22*outY + t23;
    Dtype denom = t31*outX + t32*outY + t33;
    Dtype denom2 = denom*denom;

		dTheta_tmp_diff[(9 * i) * (output_H_ * output_W_ * C) + idx] += delta_dpx * outX/denom;
		dTheta_tmp_diff[(9 * i + 1) * (output_H_ * output_W_ * C) + idx] += delta_dpx * outY/denom;
		dTheta_tmp_diff[(9 * i + 2) * (output_H_ * output_W_ * C) + idx] += delta_dpx/denom;
		dTheta_tmp_diff[(9 * i + 3) * (output_H_ * output_W_ * C) + idx] += delta_dpy * outX/denom;
		dTheta_tmp_diff[(9 * i + 4) * (output_H_ * output_W_ * C) + idx] += delta_dpy * outY/denom;
    dTheta_tmp_diff[(9 * i + 5) * (output_H_ * output_W_ * C) + idx] += delta_dpy/denom;

    dTheta_tmp_diff[(9 * i + 6) * (output_H_ * output_W_ * C) + idx] += delta_dpx * ( nomX*(-1)*(1.0/denom2)*outX );
    dTheta_tmp_diff[(9 * i + 6) * (output_H_ * output_W_ * C) + idx] += delta_dpy * ( nomY*(-1)*(1.0/denom2)*outX );

		dTheta_tmp_diff[(9 * i + 7) * (output_H_ * output_W_ * C) + idx] += delta_dpx * ( nomX*(-1)*(1.0/denom2)*outY );
		dTheta_tmp_diff[(9 * i + 7) * (output_H_ * output_W_ * C) + idx] += delta_dpy * ( nomY*(-1)*(1.0/denom2)*outY );

		dTheta_tmp_diff[(9 * i + 8) * (output_H_ * output_W_ * C) + idx] += delta_dpx * ( nomX*(-1)*(1.0/denom2) );
		dTheta_tmp_diff[(9 * i + 8) * (output_H_ * output_W_ * C) + idx] += delta_dpy * ( nomY*(-1)*(1.0/denom2) );
	}
}

template <typename Dtype>
void ProjectiveTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "ProjectiveTransformerLayer::Backward_GPU::\t";

//caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_gpu_diff());
//caffe_gpu_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_gpu_diff());
//return;

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid.gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();

	caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);

	const int nthreads = N * C * output_H_ * output_W_;
	ProjectiveTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS,0,Caffe::cuda_stream()>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
					dV, U, dTheta_tmp_diff, bottom[1]->gpu_data());

	Dtype* all_ones_2_data = all_ones_2.mutable_gpu_data();
	caffe_gpu_set(all_ones_2.count(), (Dtype)1., all_ones_2_data);
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, bottom[1]->count(), 1, output_H_ * output_W_ * C,
			(Dtype)1., dTheta_tmp_diff, all_ones_2_data, (Dtype)0., dTheta);

	/*const Dtype* db_dfull_theta = full_theta.cpu_diff();
	for(int i=0; i<full_theta.count(); ++i) {
		std::cout << db_dFull_theta[i] << " ";
	}
	std::cout<<std::endl;*/
			
	/*int k = 0;
	const int num_threads = N;
	for(int i=0; i<6; ++i) {
    copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads), CAFFE_CUDA_NUM_THREADS>>>(num_threads,
      6, i, dFull_theta, 6, k, dTheta);
    ++ k;
	}*/
	
	/*const Dtype* db_dtheta = bottom[1]->cpu_diff();
	for(int i=0; i<bottom[1]->count(); ++i) {
		std::cout << db_dtheta[i] << " ";
	}
	std::cout<<std::endl;*/
			
}

INSTANTIATE_LAYER_GPU_FUNCS(ProjectiveTransformerLayer);

}	// namespace caffe
