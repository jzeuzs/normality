# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0](https://github.com/jzeuzs/normality/compare/v1.1.0...v2.0.0) - 2025-12-08

### Added

- [**breaking**] mark the `Error` type as `non_exhaustive` ([#26](https://github.com/jzeuzs/normality/pull/26))
- implement energy test for normality ([#24](https://github.com/jzeuzs/normality/pull/24))

### Fixed

- *(deps)* remove unused dependency `gauss-quad`
- *(deps)* update rust crate ndarray to 0.17.0 ([#20](https://github.com/jzeuzs/normality/pull/20))

### Other

- version bump ([#27](https://github.com/jzeuzs/normality/pull/27))
- *(deps)* update actions/checkout action to v6 ([#23](https://github.com/jzeuzs/normality/pull/23))
- *(deps)* update rust crate pastey to 0.2.0 ([#22](https://github.com/jzeuzs/normality/pull/22))

## [1.1.0](https://github.com/jzeuzs/normality/compare/v1.0.0...v1.1.0) - 2025-11-01

### Added

- implement the Anscombe-Glynn test ([#19](https://github.com/jzeuzs/normality/pull/19))
- change `data` type on methods to `impl IntoIterator` ([#7](https://github.com/jzeuzs/normality/pull/7))

### Fixed

- *(ci)* replace mock with actual bench ([#15](https://github.com/jzeuzs/normality/pull/15))
- *(ci)* benchmark threshold ([#14](https://github.com/jzeuzs/normality/pull/14))

### Other

- *(release)* version bump
- *(bench)* rm thresholds
- use `sort_unstable_by` ([#18](https://github.com/jzeuzs/normality/pull/18))
- *(deps)* update dawidd6/action-download-artifact action to v11 ([#17](https://github.com/jzeuzs/normality/pull/17))
- *(deps)* update actions/upload-artifact action to v5 ([#16](https://github.com/jzeuzs/normality/pull/16))
- *(deps)* update actions/github-script action to v8 ([#12](https://github.com/jzeuzs/normality/pull/12))
- *(bench)* add pr threshold ([#13](https://github.com/jzeuzs/normality/pull/13))
- *(deps)* update actions/checkout action to v5 ([#11](https://github.com/jzeuzs/normality/pull/11))
- add implementation benchmarks ([#10](https://github.com/jzeuzs/normality/pull/10))
- add windows test ([#9](https://github.com/jzeuzs/normality/pull/9))

## [1.0.0](https://github.com/jzeuzs/normality/releases/tag/v1.0.0) - 2025-10-19

### Added

- inital implementation

### Other

- use `runner.temp`
- install R
- temp
- Merge branch 'main' of https://github.com/jzeuzs/normality
- remove artifact
- Add renovate.json ([#1](https://github.com/jzeuzs/normality/pull/1))
- fix temporary directory
- fix tempfile
- Initial commit
