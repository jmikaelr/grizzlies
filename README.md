<p align="center">
  <img src="assets/grizzlies-banner.png" alt="Grizzlies Logo" width="400"/>
</p>

# 🐻🔥 Grizzlies

**Grizzlies** is an experimental **dataframe library** written in [Mojo](https://www.modular.com/mojo).  
It aims to provide a **Pandas/Polars-like API** with the **raw speed and systems-level control** of Mojo, designed for modern data science, analytics, and high-performance computing.

At its core, Grizzlies is built around the `Series` struct — a 1D labeled array holding data of a single type, with efficient memory management and formatting utilities.

---

## ✨ Features (Current)


---

## 📚 Example Usage

## 🗺️ Roadmap

### Core Structures
- [ ] `DataFrame` built on top of `Series` collections  
- [ ] Column and row indexing (`loc`, `iloc` equivalents)  
- [ ] Support for common dtypes (ints, floats, bools, strings)  

### IO & Interop
- [ ] CSV and Parquet readers/writers  
- [ ] Python interoperability (via Python bindings)  
- [ ] Export to/from Arrow format  

### Operations
- [ ] Basic arithmetic between Series/DataFrames  
- [ ] GroupBy and aggregation  
- [ ] Join/merge operations  
- [ ] Reshaping (pivot, melt, stack, unstack)  

### Performance
- [ ] Parallelized execution using Mojo’s concurrency model  
- [ ] SIMD/vectorized operations  
- [ ] GPU-accelerated kernels for heavy workloads  

### Quality
- [ ] Unit testing framework  
- [ ] Benchmarks vs Pandas/Polars  
- [ ] Documentation and tutorials  

---

## 🔧 Installation

Currently experimental. Clone the repository and build with [pixi](https://prefix.dev/docs/pixi):

```bash
git clone https://github.com/jmikaelr/grizzlies
```

---

## 🤝 Contributing

We welcome contributions!  
- File issues and feature requests on GitHub.  
- Submit PRs with clear commit messages.  
- Follow coding style and keep performance in mind.  

---

## 📜 License

[MIT](LICENSE)

---

## 🌌 Acknowledgements

- Inspired by [Pandas](https://pandas.pydata.org/) and [Polars](https://www.pola.rs/).  
- Mascot concept: **Grizzly bear cub + Mojo flame buddy**.
