// SPDX-License-Identifier: Apache-2.0
//! L2 bindings (ad/window-L2-bindings): P8's rule table (`rules.rs`: kernel_class, command_class), its plan
//! view and its copy-row form (`rows.rs`: rows, gather) compiled into the P1 runner unchanged, so the per-launch
//! dot (addot.rs) and the composed dispatcher read one table. The three files are P8's own (tree/ad/phys/p8/src).
#[allow(dead_code, unused)] #[path = "../../../../../p8/src/plan.rs"] pub mod plan;
#[allow(dead_code, unused)] #[path = "../../../../../p8/src/rules.rs"] pub mod rules;
#[allow(dead_code, unused)] #[path = "../../../../../p8/src/rows.rs"] pub mod rows;
