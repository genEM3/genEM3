class DataWriter:

    def __init__(self,
                 dataloader,
                 output_collate_fn,
                 output_label,
                 output_path,
                 output_dtype):
        self.dataloader = dataloader
        self.output_collate_fn = output_collate_fn
        self.output_label = output_label
        self.output_path = output_path
        self.output_dtype = output_dtype

    def batch_to_cache(self, outputs, output_inds):
        self.dataloader.dataset.write_output_to_cache(self.output_collate_fn(outputs), output_inds, self.output_label,
                                                      self.output_dtype)

    def cache_to_wkw(self):
        pass