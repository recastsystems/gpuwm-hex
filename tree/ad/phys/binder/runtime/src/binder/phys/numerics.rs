// SPDX-License-Identifier: Apache-2.0
//! Derivative compilation options for the executed forward numerics route.
//!
//! The binder does not compile derivatives yet. Integration must identify the
//! route from the pinned forward image and record both images' actual options.
//! These helpers do not modify or rebuild the ordinary-forward image.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardNumerics { Strict, Fast }

impl ForwardNumerics {
    pub fn parse(route: &str) -> Result<Self, &'static str> {
        match route {
            "strict" => Ok(Self::Strict),
            "fast" => Ok(Self::Fast),
            _ => Err("unknown forward numerics route"),
        }
    }
    pub fn derivative_ftz(self) -> &'static str {
        match self { Self::Strict => "--ftz=false", Self::Fast => "--ftz=true" }
    }
}

/// Preserve every existing option and append the matching denormal mode.
/// Conflicting or ambiguous explicit mode flags fail rather than silently
/// changing the source route or the caller's remaining compiler options.
pub fn derivative_options(route: ForwardNumerics, options: &[String]) -> Result<Vec<String>, &'static str> {
    let expected = route.derivative_ftz().strip_prefix("--ftz=").unwrap();
    let mut present = false;
    for option in options {
        if option == "--ftz" || option == "-ftz" { return Err("ambiguous ftz option"); }
        if let Some(value) = option.strip_prefix("--ftz=").or_else(|| option.strip_prefix("-ftz=")) {
            if value != expected { return Err("derivative ftz conflicts with forward route"); }
            present = true;
        }
        if matches!(option.as_str(), "--use_fast_math" | "-use_fast_math") && route == ForwardNumerics::Strict {
            return Err("fast math conflicts with strict forward denormal mode");
        }
    }
    let mut result = options.to_vec();
    if !present { result.push(route.derivative_ftz().into()); }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn each_route_preserves_options_and_adds_matching_mode() {
        let input = vec!["--std=c++17".to_string()];
        assert_eq!(derivative_options(ForwardNumerics::Strict, &input).unwrap(),
                   ["--std=c++17", "--ftz=false"]);
        assert_eq!(derivative_options(ForwardNumerics::Fast, &input).unwrap(),
                   ["--std=c++17", "--ftz=true"]);
        assert!(ForwardNumerics::parse("unknown").is_err());
    }
    #[test]
    fn conflicting_or_ambiguous_modes_are_rejected() {
        for option in ["--ftz=true", "-ftz=true", "--use_fast_math", "--ftz"] {
            assert!(derivative_options(ForwardNumerics::Strict, &[option.into()]).is_err());
        }
        assert!(derivative_options(ForwardNumerics::Fast, &["--ftz=false".into()]).is_err());
        assert_eq!(derivative_options(ForwardNumerics::Strict, &["--ftz=false".into()]).unwrap(),
                   ["--ftz=false"]);
    }
}
