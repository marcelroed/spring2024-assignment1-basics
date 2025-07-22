use pyo3_build_config;

fn main() {
    if cfg!(unix) {
        if let Some(lib_dir) = &pyo3_build_config::get().lib_dir {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir);
        } else {
            panic!("pyo3_build_config::get().lib_dir is None, this should not happen");
        }
    }
}