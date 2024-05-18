#ifndef DATA_LOADER_H
#define DATA_LOADER_H



struct prefetch_params
{
	uint32_t batch_size;
	uint64_t starting_index;
	void* data_loader;
};

template <typename Dtype>
int prefetch_threadproc(void* params, int block_index, int total_blocks);

template <typename Dtype>
class DataLoader
{
public:
	DataLoader()
	{
		batch_size_ = 0;
		total_examples_ = 0;
		index_ = 0;
		eof_ = false;
		prefetched_ = false;
	}

	~DataLoader() {}

	bool Init(uint32_t batch_size, uint64_t total_examples, uint32_t thread_count, Dtype* training_examples)
	{
		batch_size_ = batch_size;
		total_examples_ = total_examples;
		thread_count_ = thread_count;
		training_examples_ = training_examples;
		index_ = 0;
		eof_ = false;
		prefetched_ = false;

		return thread_pool_.Init(thread_count);
	}

	Dtype* GetBatch()
	{
		int* ret;

		if (!prefetched_)
		{
			assert(0);
		}

		thread_pool_.WaitForTaskCompletion(&ret);

		if (eof_)
		{
			index_ = 0;
			eof_ = false;
			prefetched_ = false;
			return nullptr;
		}

		if (index_ + batch_size_ < total_examples_)
		{
			index_ += batch_size_;
		}
		else
		{
			eof_ = true;
			index_ = total_examples_;
		}

		prefetched_ = false;

		return training_examples_;
	}

	void Prefetch()
	{
		int32_t batch_size;
		static prefetch_params pf_params;

		prefetched_ = true;
		//batch_size = std::min((int)batch_size_, (int)(total_examples_ - index_));
		if (batch_size_ < (total_examples_ - index_))
		{
			batch_size = batch_size_;
		}
		else
		{
			batch_size = (int32_t)(total_examples_ - index_);
		}

		if (batch_size > 0)
		{
			set_batch_size(batch_size); // don't call set_batch_size when batch_size == 0 (happens at end of training set, i.e. eof state)
		}
		else
		{
			assert(batch_size == 0); // should be zero if all book keeping is bug free
		}


		pf_params.batch_size = batch_size;
		pf_params.starting_index = index_;
		pf_params.data_loader = this;

		thread_pool_.Execute(prefetch_threadproc<Dtype>, &pf_params, thread_pool_.get_thread_count());
	}

	void Randomize()
	{
		uint64_t index_1;
		uint64_t index_2;
		uint64_t i;

		srand(888);

		for (i = 0; i < total_examples_ * 2; i++)
		{
			while (true)
			{
				index_1 = (rand() + (rand() << 15)) % total_examples_;
				index_2 = (rand() + (rand() << 15)) % total_examples_;

				if (index_1 != index_2)
				{
					break;
				}
			}
			swap_examples(index_1, index_2);
		}
	}

	virtual void get_item(uint64_t training_set_index, uint32_t batch_index) = 0;
	virtual void set_batch_size(uint32_t new_batch_size) = 0;
	virtual void swap_examples(uint64_t index_1, uint64_t index_2) = 0;

private:
	uint32_t batch_size_;
	uint64_t total_examples_;
	uint32_t thread_count_;
	uint64_t index_;
	Dtype* training_examples_;
	bool eof_;
	bool prefetched_;
	ThreadPool thread_pool_;
};


template <typename Dtype>
int prefetch_threadproc(void* params, int block_index, int total_blocks)
{
	int i;
	int batch_size;

	prefetch_params* pf_params = (prefetch_params*)params;

	DataLoader<Dtype>* dl = (DataLoader<Dtype>*)pf_params->data_loader;
	batch_size = pf_params->batch_size;

	for (i = block_index; i < batch_size; i += total_blocks)
	{
		dl->get_item(pf_params->starting_index + i, i);
	}

	return 0;
}


#endif //DATA_LOADER_H
