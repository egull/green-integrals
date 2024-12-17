/*
 * Copyright (c) 2023 University of Michigan.
 *
 */

#ifndef GREEN_DFINTEGRAL_H
#define GREEN_DFINTEGRAL_H

#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>

#include "df_buffered_reader.h"
#include "common_defs.h"

namespace green::integrals{


  /**
   * @brief Integral class to parse Density fitted 3-center integrals, handles reading given by the path argument
   */
  class df_integral_t {
    // prefixes for hdf5
    const std::string _chunk_basename    = "VQ";
    const std::string _corr_path         = "df_ewald.h5";
    const std::string _corr_basename     = "EW";
    const std::string _corr_bar_basename = "EW_bar";

    using bz_utils_t                     = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using int_data                       = utils::shared_object<ztensor<4>>;

  public:
    df_integral_t(const std::string& path, int nao, int NQ, const bz_utils_t& bz_utils) :
      _base_path(path),
      _number_of_keys(bz_utils.symmetry().num_kpair_stored()),
      _vij_Q_buffer(path, nao, NQ, _number_of_keys, 0.5), //initialize buffered reader
        _k0(-1), _NQ(NQ), _bz_utils(bz_utils) {
    }

    virtual ~df_integral_t() {}

    void read_integrals(size_t k1, size_t k2){
        ;
    }
    template <typename type>
    void read_entire(std::complex<type>* Vk1k2_Qij, int intranode_rank, int intranode_size) {
      std::array<size_t, 4>  shape=_vij_Q_buffer.shape();
      std::size_t NQ=shape[1];
      std::size_t nao=shape[2];
      std::size_t element_size=NQ*nao*nao;

      //access every element. This will read it.
      std::size_t this_rank_startindex=_number_of_keys/intranode_size*intranode_rank;
      std::size_t this_rank_endindex=std::min(_number_of_keys/intranode_size*(intranode_rank+1), _number_of_keys);

      std::cout<<"full integral read. rank : "<<intranode_rank<<" start: "<<this_rank_startindex<<" end: "<<this_rank_endindex<<std::endl;

      for(std::size_t key=this_rank_startindex;key<this_rank_endindex;key++){
        const std::complex<double> *buffer=_vij_Q_buffer.access_element(key);
        std::size_t offset=element_size*key;
        _vij_Q_buffer.release_element(key);
        Complex_DoubleToType(buffer, Vk1k2_Qij+offset, NQ*nao*nao);
      }
    }


    void Complex_DoubleToType(const std::complex<double>* in, std::complex<double>* out, size_t size) {
      memcpy(out, in, size * sizeof(std::complex<double>));
    }

    void Complex_DoubleToType(const std::complex<double>* in, std::complex<float>* out, size_t size) {
      for (int i = 0; i < size; ++i) {
        out[i] = static_cast<std::complex<float>>(in[i]);
      }
    }

    /**
     * read next part of the G=0 correction to interaction integral for the specific k-point
     * @param file - file to be used
     * @param k - k-point
     */
    void read_correction(int k) {
      //auto shape = _vij_Q.shape();
      auto shape = _vij_Q_buffer.shape();
      _v0ij_Q.resize(shape[1], shape[2], shape[3]);
      _v_bar_ij_Q.resize(shape[1], shape[2], shape[3]);
      // avoid unnecessary reading
      if (k == _k0) {
        // we have data cached
        return;
      }
      _k0                 = k;
      std::string   fname = _base_path + "/" + _corr_path;
      h5pp::archive ar(fname);
      // Construct integral dataset name
      std::string   dsetnum = _corr_basename + "/" + std::to_string(k);
      // read data
      ar[dsetnum] >> reinterpret_cast<double*>(_v0ij_Q.data());
      // Construct integral dataset name
      dsetnum = _corr_bar_basename + "/" + std::to_string(k);
      // read data
      ar[dsetnum] >> reinterpret_cast<double*>(_v_bar_ij_Q.data());
      ar.close();
    };

    /**
     * Determine the type of symmetries for the integral based on the current k-points
     *
     * @param k1 incomming k-point
     * @param k2 outgoing k-point
     * @return A pair of sign and type of applied symmetry
     */
    std::pair<int, integral_symmetry_type_e> v_type(size_t k1, size_t k2) {
      size_t idx  = momenta_to_key(k1,k2);
      // determine sign
      int    sign = (k1 >= k2) ? 1 : -1;
      // determine applied symmetry type
      // by default no symmetries applied
      integral_symmetry_type_e symmetry_type = direct;
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        symmetry_type = conjugated;
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        symmetry_type = transposed;
      }
      return std::make_pair(sign, symmetry_type);
    }


    /**
     * Extract V(Q, i, j) with given (k1, k2) in precision "prec" from the entire integrals (Vk1k2_Qij)
     * TODO: this non-chunked version should be combined with the chunked version
     * @param Vk1k2_Qij
     * @param V
     * @param k1
     * @param k2
     */
    template <typename prec>
    void symmetrize(const std::complex<double>* Vk1k2_Qij, tensor<prec, 3>& V, const int k1, const int k2) {
      int                                      key = momenta_to_symmred_key(k1, k2);
      std::pair<int, integral_symmetry_type_e> vtype            = v_type(k1, k2);
      size_t                                   NQ               = V.shape()[0];
      size_t                                   nao              = V.shape()[1];
      size_t                                   shift            = key * NQ * nao * nao;
      size_t                                   element_counts_V = NQ * nao * nao;
      ztensor<3>                               V_double_buffer(NQ, nao, nao);
      memcpy(V_double_buffer.data(), Vk1k2_Qij + shift, element_counts_V * sizeof(std::complex<double>));
      if (vtype.first < 0) {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V_double_buffer(Q)).transpose().conjugate().eval().cast<prec>();
        }
      } else {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V_double_buffer(Q)).cast<prec>();
        }
      }
      if (vtype.second == conjugated) {  // conjugate
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V(Q)).conjugate();
        }
      } else if (vtype.second == transposed) {  // transpose
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V(Q)).transpose().eval();
        }
      }
    }


    /**
     * Extract V(Q, i, j) with given (k1, k2) from chunks of integrals (_vij_Q)
     * Note that Q here denotes the auxiliary basis index, not the transfer momentum
     * Also apply conjugate transpose, conjugate, or transpose.
     * @tparam prec
     * @param vij_Q_k1k2
     * @param k1
     * @param k2
     */
    template <typename prec>
    void symmetrize(tensor<prec, 3>& vij_Q_k1k2, size_t k1, size_t k2, size_t NQ_offset = 0, size_t NQ_local = 0) {
      std::pair<int, integral_symmetry_type_e> vtype     = v_type(k1, k2);
      int                                      NQ        = _NQ;
      NQ_local                                           = (NQ_local == 0) ? NQ : NQ_local;
      int nao=_vij_Q_buffer.nao();
      //int nao=_vij_Q.shape()[2];
      int key=momenta_to_symmred_key(k1,k2);
      typedef Eigen::Map<const Eigen::Matrix<std::complex<double>,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> map_t;
      const std::complex<double> *elem_ptr=_vij_Q_buffer.access_element(key);
      if (vtype.first < 0) {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          map_t vijb_map(elem_ptr+Q*nao*nao,nao,nao);
          matrix(vij_Q_k1k2(Q_loc)) = vijb_map.transpose().conjugate().cast<prec>();
        }
      } else {
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          map_t vijb_map(elem_ptr+Q*nao*nao,nao,nao);
          matrix(vij_Q_k1k2(Q_loc)) = vijb_map.cast<prec>();
        }
      }
      _vij_Q_buffer.release_element(key);
      if (vtype.second == conjugated) {  // conjugate
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q_k1k2(Q_loc)).conjugate();

        }
      } else if (vtype.second == transposed) {  // transpose
        for (int Q = NQ_offset, Q_loc = 0; Q_loc < NQ_local; ++Q, ++Q_loc) {
          matrix(vij_Q_k1k2(Q_loc)) = matrix(vij_Q_k1k2(Q_loc)).transpose().eval();
        }
      }
    }


    //const ztensor<4>& vij_Q() const { return _vij_Q()->object(); }
    const ztensor<3>& v0ij_Q() const { return _v0ij_Q; }
    const ztensor<3>& v_bar_ij_Q() const { return _v_bar_ij_Q; }
    const std::complex<double> *access_vij_Q(int k1, int k2) //not const because we're modifying buffers inside vij
    {
      int key=momenta_to_symmred_key(k1,k2);
      return _vij_Q_buffer.access_element(key);
    }
    void release_vij_Q(int k1, int k2){
      int key=momenta_to_symmred_key(k1,k2);
      _vij_Q_buffer.release_element(key);
    }

    int momenta_to_key(int k1, int k2) const{
      size_t idx = (k1 >= k2) ? k1 * (k1 + 1) / 2 + k2 : k2 * (k2 + 1) / 2 + k1;  // k-pair = (k1, k2) or (k2, k1)
      return idx;
    }
    int momenta_to_symmred_key(int k1, int k2) const{
      int idx=momenta_to_key(k1,k2);
      // determine type
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      int idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      return idx_red;
    }

    void reset() {
      _vij_Q_buffer.reset();
    }
    std::size_t shape() const{
      auto shape = _vij_Q_buffer.shape();
      return shape[0]*shape[1]*shape[2]*shape[3]; //nkeys*nQ*nao*nao
    }

  private:
    int                       _number_of_keys;
    //df_legacy_reader _vij_Q;
    df_buffered_reader _vij_Q_buffer;
    // G=0 correction to coulomb integral stored in density fitting format for second-order e3xchange diagram
    ztensor<3>                _v0ij_Q;
    ztensor<3>                _v_bar_ij_Q;

    bool                      _exch;
    // current leading index
    int                       _k0;
    long                      _NQ;
    const bz_utils_t&         _bz_utils;

    // base path to integral files
    std::string               _base_path;
  };

}  // namespace green::integrals

#endif  // GF2_DFINTEGRAL_H
