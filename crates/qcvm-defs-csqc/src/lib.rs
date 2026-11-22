//! # Client-side QuakeC (CSQC) definitions for `qcvm`
//!
//! CSQC (`csprogs.dat`) lets mods draw the HUD (and, for "full" CSQC, the whole view) from QuakeC,
//! as supported by FTEQW, DarkPlaces and QuakeSpasm-Spiked. See
//! <https://quakewiki.org/wiki/EXT_CSQC>.
//!
//! [`Csqc`] is the defs of "simple CSQC" `csprogs.dat`s, which are compiled against the same system
//! globals and fields as vanilla Quake 1 (`progdefs.q1`, CRC 5927), as QuakeSpasm-Spiked uses. FTEQW
//! and DarkPlaces' full CSQC defs (CRCs 22390 and 52195) have their own system globals and aren't
//! supported yet.
//!
//! The builtins ([`BuiltinAddr`] and [`builtins::NAMED_BUILTINS`]) are the core CSQC builtins:
//! those that both FTEQW and QuakeSpasm-Spiked support in CSQC. Builtins that are shared with
//! Quake 1 have the same canonical names as in `qcvm_defs_quake1`.

#![deny(missing_docs)]

pub mod builtins;

pub use self::builtins::BuiltinAddr;
pub use qcvm_defs_quake1::{FieldAddr, GlobalAddr};

use qcvm::QcDefs;

/// The defs of "simple CSQC" `csprogs.dat`s: the Quake 1 system globals and fields, with the core
/// CSQC builtins.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Csqc;

impl QcDefs for Csqc {
    type GlobalAddr = GlobalAddr;
    type FieldAddr = FieldAddr;
    type BuiltinAddr = BuiltinAddr;

    const CRC: i32 = 5927;
}

/// The API level that hosts should pass to `CSQC_Init`.
///
/// Mods may compare this against the level they need. Hosts that implement more of the CSQC API
/// can pass a higher level.
pub const CSQC_API_VERSION: f32 = 1.;

/// The names of the functions that CSQC can define, which the engine calls.
pub mod entry_points {
    /// `void(float apilevel, string enginename, float engineversion)`: called when the
    /// `csprogs.dat` is loaded.
    pub const INIT: &str = "CSQC_Init";
    /// `void()`: called before the `csprogs.dat` is unloaded.
    pub const SHUTDOWN: &str = "CSQC_Shutdown";
    /// `void(float vwidth, float vheight, float notmenu)`: draws the whole view ("full" CSQC).
    /// If this is defined, the engine draws nothing itself, and [`DRAW_HUD`] and [`DRAW_SCORES`]
    /// aren't called.
    pub const UPDATE_VIEW: &str = "CSQC_UpdateView";
    /// `void(vector viewsize, float scoresshown)`: draws the HUD after the engine draws the 3D
    /// view ("simple" CSQC).
    pub const DRAW_HUD: &str = "CSQC_DrawHud";
    /// `void(vector viewsize, float scoresshown)`: draws the scoreboard when it is shown
    /// ("simple" CSQC).
    pub const DRAW_SCORES: &str = "CSQC_DrawScores";
    /// `float(float evtype, float scanx, float chary, float devid)`: called for input events.
    /// Returns `TRUE` if the event was handled, in which case the engine ignores it.
    pub const INPUT_EVENT: &str = "CSQC_InputEvent";
    /// `float(string cmd)`: called for console commands registered with `registercommand`.
    /// Returns `TRUE` if the command was handled.
    pub const CONSOLE_COMMAND: &str = "CSQC_ConsoleCommand";
    /// `float(string msg)`: called for centerprints from the server. Returns `TRUE` if the
    /// centerprint was handled, in which case the engine doesn't show it.
    pub const PARSE_CENTER_PRINT: &str = "CSQC_Parse_CenterPrint";
    /// `void(string msg, float printlvl)`: called for prints from the server, which CSQC is then
    /// responsible for showing.
    pub const PARSE_PRINT: &str = "CSQC_Parse_Print";
    /// `float(float save, float take, vector inflictororg)`: called when the player takes damage.
    /// Returns `TRUE` if the damage was handled, in which case the engine doesn't show its own
    /// effects.
    pub const PARSE_DAMAGE: &str = "CSQC_Parse_Damage";
}
