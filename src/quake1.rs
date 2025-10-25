//! # Quake field, global and builtin definitions
//!
//! > TODO: Docs.

pub use globals::GLOBALS_RANGE;

pub mod globals {
    use std::ops::Range;

    use strum::EnumIter;

    use crate::progs::{ScalarType, Type};

    pub const GLOBALS_START: u32 = 28;

    /// Seismon has "true" locals and does not reuse globals for locals, for correctness and
    /// resiliency.
    pub const LOCALS_START: u32 = GlobalAddr::SetChangeArgs.to_u16() as u32 + 1;

    pub const GLOBALS_RANGE: Range<usize> = GLOBALS_START as usize..LOCALS_START as usize;

    #[derive(EnumIter, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum GlobalAddr {
        Self_,             // entity self
        Other,             // entity other
        World,             // entity world
        Time,              // float  time
        FrameTime,         // float  frametime
        ForceRetouch,      // float  force_retouch
        MapName,           // string mapname
        Deathmatch,        // float  deathmatch
        Coop,              // float  coop
        TeamPlay,          // float  teamplay
        ServerFlags,       // float  serverflags
        TotalSecrets,      // float  total_secrets
        TotalMonsters,     // float  total_monsters
        FoundSecrets,      // float  found_secrets
        KilledMonsters,    // float  killed_monsters
        Arg0,              // float  parm1
        Arg1,              // float  parm2
        Arg2,              // float  parm3
        Arg3,              // float  parm4
        Arg4,              // float  parm5
        Arg5,              // float  parm6
        Arg6,              // float  parm7
        Arg7,              // float  parm8
        Arg8,              // float  parm9
        Arg9,              // float  parm10
        Arg10,             // float  parm11
        Arg11,             // float  parm12
        Arg12,             // float  parm13
        Arg13,             // float  parm14
        Arg14,             // float  parm15
        Arg15,             // float  parm16
        VForward,          // vector v_forward
        VForwardX,         // float v_forward_x
        VForwardY,         // float v_forward_y
        VForwardZ,         // float v_forward_z
        VUp,               // vector v_up
        VUpX,              // float v_up_x
        VUpY,              // float v_up_y
        VUpZ,              // float v_up_z
        VRight,            // vector v_right
        VRightX,           // float v_right_x
        VRightY,           // float v_right_y
        VRightZ,           // float v_right_z
        TraceAllSolid,     // float  trace_allsolid
        TraceStartSolid,   // float  trace_startsolid
        TraceFraction,     // float  trace_fraction
        TraceEndPos,       // vector trace_endpos
        TraceEndPosX,      // float trace_endpos_x
        TraceEndPosY,      // float trace_endpos_y
        TraceEndPosZ,      // float trace_endpos_z
        TracePlaneNormal,  // vector trace_plane_normal
        TracePlaneNormalX, // float trace_plane_normal_x
        TracePlaneNormalY, // float trace_plane_normal_y
        TracePlaneNormalZ, // float trace_plane_normal_z
        TracePlaneDist,    // float  trace_plane_dist
        TraceEntity,       // entity trace_ent
        TraceInOpen,       // float  trace_inopen
        TraceInWater,      // float  trace_inwater
        MsgEntity,         // entity msg_entity

        // Functions
        Main,              // function main
        StartFrame,        // function StartFrame
        PlayerPreThink,    // function PlayerPreThink
        PlayerPostThink,   // function PlayerPostThink
        ClientKill,        // function ClientKill
        ClientConnect,     // function ClientConnect
        PutClientInServer, // function PutClientInServer
        ClientDisconnect,  // function ClientDisconnect
        SetNewArgs,        // function SetNewParms
        SetChangeArgs,     // function SetChangeParms
    }

    impl GlobalAddr {
        pub const fn from_u16(val: u16, ty: Type) -> Option<Self> {
            match (val, ty) {
                (28, Type::Scalar(ScalarType::Entity)) => Some(Self::Self_),
                (29, Type::Scalar(ScalarType::Entity)) => Some(Self::Other),
                (30, Type::Scalar(ScalarType::Entity)) => Some(Self::World),
                (31, Type::Scalar(ScalarType::Float)) => Some(Self::Time),
                (32, Type::Scalar(ScalarType::Float)) => Some(Self::FrameTime),
                (33, Type::Scalar(ScalarType::Float)) => Some(Self::ForceRetouch),
                (34, Type::Scalar(ScalarType::String)) => Some(Self::MapName),
                (35, Type::Scalar(ScalarType::Float)) => Some(Self::Deathmatch),
                (36, Type::Scalar(ScalarType::Float)) => Some(Self::Coop),
                (37, Type::Scalar(ScalarType::Float)) => Some(Self::TeamPlay),
                (38, Type::Scalar(ScalarType::Float)) => Some(Self::ServerFlags),
                (39, Type::Scalar(ScalarType::Float)) => Some(Self::TotalSecrets),
                (40, Type::Scalar(ScalarType::Float)) => Some(Self::TotalMonsters),
                (41, Type::Scalar(ScalarType::Float)) => Some(Self::FoundSecrets),
                (42, Type::Scalar(ScalarType::Float)) => Some(Self::KilledMonsters),
                (43, Type::Scalar(ScalarType::Float)) => Some(Self::Arg0),
                (44, Type::Scalar(ScalarType::Float)) => Some(Self::Arg1),
                (45, Type::Scalar(ScalarType::Float)) => Some(Self::Arg2),
                (46, Type::Scalar(ScalarType::Float)) => Some(Self::Arg3),
                (47, Type::Scalar(ScalarType::Float)) => Some(Self::Arg4),
                (48, Type::Scalar(ScalarType::Float)) => Some(Self::Arg5),
                (49, Type::Scalar(ScalarType::Float)) => Some(Self::Arg6),
                (50, Type::Scalar(ScalarType::Float)) => Some(Self::Arg7),
                (51, Type::Scalar(ScalarType::Float)) => Some(Self::Arg8),
                (52, Type::Scalar(ScalarType::Float)) => Some(Self::Arg9),
                (53, Type::Scalar(ScalarType::Float)) => Some(Self::Arg10),
                (54, Type::Scalar(ScalarType::Float)) => Some(Self::Arg11),
                (55, Type::Scalar(ScalarType::Float)) => Some(Self::Arg12),
                (56, Type::Scalar(ScalarType::Float)) => Some(Self::Arg13),
                (57, Type::Scalar(ScalarType::Float)) => Some(Self::Arg14),
                (58, Type::Scalar(ScalarType::Float)) => Some(Self::Arg15),
                (59, Type::Vector) => Some(Self::VForward),
                (59, Type::Scalar(ScalarType::Float)) => Some(Self::VForwardX),
                (60, Type::Scalar(ScalarType::Float)) => Some(Self::VForwardY),
                (61, Type::Scalar(ScalarType::Float)) => Some(Self::VForwardZ),
                (62, Type::Vector) => Some(Self::VUp),
                (62, Type::Scalar(ScalarType::Float)) => Some(Self::VUpX),
                (63, Type::Scalar(ScalarType::Float)) => Some(Self::VUpY),
                (64, Type::Scalar(ScalarType::Float)) => Some(Self::VUpZ),
                (65, Type::Vector) => Some(Self::VRight),
                (65, Type::Scalar(ScalarType::Float)) => Some(Self::VRightX),
                (66, Type::Scalar(ScalarType::Float)) => Some(Self::VRightY),
                (67, Type::Scalar(ScalarType::Float)) => Some(Self::VRightZ),
                (68, Type::Scalar(ScalarType::Float)) => Some(Self::TraceAllSolid),
                (69, Type::Scalar(ScalarType::Float)) => Some(Self::TraceStartSolid),
                (70, Type::Scalar(ScalarType::Float)) => Some(Self::TraceFraction),
                (71, Type::Vector) => Some(Self::TraceEndPos),
                (71, Type::Scalar(ScalarType::Float)) => Some(Self::TraceEndPosX),
                (72, Type::Scalar(ScalarType::Float)) => Some(Self::TraceEndPosY),
                (73, Type::Scalar(ScalarType::Float)) => Some(Self::TraceEndPosZ),
                (74, Type::Vector) => Some(Self::TracePlaneNormal),
                (74, Type::Scalar(ScalarType::Float)) => Some(Self::TracePlaneNormalX),
                (75, Type::Scalar(ScalarType::Float)) => Some(Self::TracePlaneNormalY),
                (76, Type::Scalar(ScalarType::Float)) => Some(Self::TracePlaneNormalZ),
                (77, Type::Scalar(ScalarType::Float)) => Some(Self::TracePlaneDist),
                (78, Type::Scalar(ScalarType::Entity)) => Some(Self::TraceEntity),
                (79, Type::Scalar(ScalarType::Float)) => Some(Self::TraceInOpen),
                (80, Type::Scalar(ScalarType::Float)) => Some(Self::TraceInWater),
                (81, Type::Scalar(ScalarType::Entity)) => Some(Self::MsgEntity),

                // Functions
                (82, Type::Scalar(ScalarType::Function)) => Some(Self::Main),
                (83, Type::Scalar(ScalarType::Function)) => Some(Self::StartFrame),
                (84, Type::Scalar(ScalarType::Function)) => Some(Self::PlayerPreThink),
                (85, Type::Scalar(ScalarType::Function)) => Some(Self::PlayerPostThink),
                (86, Type::Scalar(ScalarType::Function)) => Some(Self::ClientKill),
                (87, Type::Scalar(ScalarType::Function)) => Some(Self::ClientConnect),
                (88, Type::Scalar(ScalarType::Function)) => Some(Self::PutClientInServer),
                (89, Type::Scalar(ScalarType::Function)) => Some(Self::ClientDisconnect),
                (90, Type::Scalar(ScalarType::Function)) => Some(Self::SetNewArgs),
                (91, Type::Scalar(ScalarType::Function)) => Some(Self::SetChangeArgs),
                _ => None,
            }
        }

        pub const fn to_u16(&self) -> u16 {
            match self {
                Self::Self_ => 28,
                Self::Other => 29,
                Self::World => 30,
                Self::Time => 31,
                Self::FrameTime => 32,
                Self::ForceRetouch => 33,
                Self::MapName => 34,
                Self::Deathmatch => 35,
                Self::Coop => 36,
                Self::TeamPlay => 37,
                Self::ServerFlags => 38,
                Self::TotalSecrets => 39,
                Self::TotalMonsters => 40,
                Self::FoundSecrets => 41,
                Self::KilledMonsters => 42,
                Self::Arg0 => 43,
                Self::Arg1 => 44,
                Self::Arg2 => 45,
                Self::Arg3 => 46,
                Self::Arg4 => 47,
                Self::Arg5 => 48,
                Self::Arg6 => 49,
                Self::Arg7 => 50,
                Self::Arg8 => 51,
                Self::Arg9 => 52,
                Self::Arg10 => 53,
                Self::Arg11 => 54,
                Self::Arg12 => 55,
                Self::Arg13 => 56,
                Self::Arg14 => 57,
                Self::Arg15 => 58,
                Self::VForward => 59,
                Self::VForwardX => 59,
                Self::VForwardY => 60,
                Self::VForwardZ => 61,
                Self::VUp => 62,
                Self::VUpX => 62,
                Self::VUpY => 63,
                Self::VUpZ => 64,
                Self::VRight => 65,
                Self::VRightX => 65,
                Self::VRightY => 66,
                Self::VRightZ => 67,
                Self::TraceAllSolid => 68,
                Self::TraceStartSolid => 69,
                Self::TraceFraction => 70,
                Self::TraceEndPos => 71,
                Self::TraceEndPosX => 71,
                Self::TraceEndPosY => 72,
                Self::TraceEndPosZ => 73,
                Self::TracePlaneNormal => 74,
                Self::TracePlaneNormalX => 74,
                Self::TracePlaneNormalY => 75,
                Self::TracePlaneNormalZ => 76,
                Self::TracePlaneDist => 77,
                Self::TraceEntity => 78,
                Self::TraceInOpen => 79,
                Self::TraceInWater => 80,
                Self::MsgEntity => 81,

                // Functions
                Self::Main => 82,
                Self::StartFrame => 83,
                Self::PlayerPreThink => 84,
                Self::PlayerPostThink => 85,
                Self::ClientKill => 86,
                Self::ClientConnect => 87,
                Self::PutClientInServer => 88,
                Self::ClientDisconnect => 89,
                Self::SetNewArgs => 90,
                Self::SetChangeArgs => 91,
            }
        }

        pub const fn name(&self) -> &str {
            match self {
                Self::Self_ => "self",
                Self::Other => "other",
                Self::World => "world",
                Self::Time => "time",
                Self::FrameTime => "frametime",
                // Self::NewMissile => "newmis",
                Self::ForceRetouch => "force_retouch",
                Self::MapName => "mapname",
                Self::Deathmatch => "deathmatch",
                Self::Coop => "coop",
                Self::TeamPlay => "teamplay",
                Self::ServerFlags => "serverflags",
                Self::TotalSecrets => "total_secrets",
                Self::TotalMonsters => "total_monsters",
                Self::FoundSecrets => "found_secrets",
                Self::KilledMonsters => "killed_monsters",
                Self::Arg0 => "parm1",
                Self::Arg1 => "parm2",
                Self::Arg2 => "parm3",
                Self::Arg3 => "parm4",
                Self::Arg4 => "parm5",
                Self::Arg5 => "parm6",
                Self::Arg6 => "parm7",
                Self::Arg7 => "parm8",
                Self::Arg8 => "parm9",
                Self::Arg9 => "parm10",
                Self::Arg10 => "parm11",
                Self::Arg11 => "parm12",
                Self::Arg12 => "parm13",
                Self::Arg13 => "parm14",
                Self::Arg14 => "parm15",
                Self::Arg15 => "parm16",
                Self::VForward => "v_forward",
                Self::VForwardX => "v_forward_x",
                Self::VForwardY => "v_forward_y",
                Self::VForwardZ => "v_forward_z",
                Self::VUp => "v_up",
                Self::VUpX => "v_up_x",
                Self::VUpY => "v_up_y",
                Self::VUpZ => "v_up_z",
                Self::VRight => "v_right",
                Self::VRightX => "v_right_x",
                Self::VRightY => "v_right_y",
                Self::VRightZ => "v_right_z",
                Self::TraceAllSolid => "trace_allsolid",
                Self::TraceStartSolid => "trace_startsolid",
                Self::TraceFraction => "trace_fraction",
                Self::TraceEndPos => "trace_endpos",
                Self::TraceEndPosX => "trace_endpos_x",
                Self::TraceEndPosY => "trace_endpos_y",
                Self::TraceEndPosZ => "trace_endpos_z",
                Self::TracePlaneNormal => "trace_plane_normal",
                Self::TracePlaneNormalX => "trace_plane_normal_x",
                Self::TracePlaneNormalY => "trace_plane_normal_y",
                Self::TracePlaneNormalZ => "trace_plane_normal_z",
                Self::TracePlaneDist => "trace_plane_dist",
                Self::TraceEntity => "trace_ent",
                Self::TraceInOpen => "trace_inopen",
                Self::TraceInWater => "trace_inwater",
                Self::MsgEntity => "msg_entity",
                Self::Main => "main",
                Self::StartFrame => "StartFrame",
                Self::PlayerPreThink => "PlayerPreThink",
                Self::PlayerPostThink => "PlayerPostThink",
                Self::ClientKill => "ClientKill",
                Self::ClientConnect => "ClientConnect",
                Self::PutClientInServer => "PutClientInServer",
                Self::ClientDisconnect => "ClientDisconnect",
                Self::SetNewArgs => "SetNewParms",
                Self::SetChangeArgs => "SetChangeParms",
            }
        }

        pub fn from_name(name: &str) -> Option<Self> {
            match name {
                "self" => Some(Self::Self_),
                "other" => Some(Self::Other),
                "world" => Some(Self::World),
                "time" => Some(Self::Time),
                "frametime" => Some(Self::FrameTime),
                // "newmis" => Some(Self::NewMissile),
                "force_retouch" => Some(Self::ForceRetouch),
                "mapname" => Some(Self::MapName),
                "deathmatch" => Some(Self::Deathmatch),
                "coop" => Some(Self::Coop),
                "teamplay" => Some(Self::TeamPlay),
                "serverflags" => Some(Self::ServerFlags),
                "total_secrets" => Some(Self::TotalSecrets),
                "total_monsters" => Some(Self::TotalMonsters),
                "found_secrets" => Some(Self::FoundSecrets),
                "killed_monsters" => Some(Self::KilledMonsters),
                "parm1" => Some(Self::Arg0),
                "parm2" => Some(Self::Arg1),
                "parm3" => Some(Self::Arg2),
                "parm4" => Some(Self::Arg3),
                "parm5" => Some(Self::Arg4),
                "parm6" => Some(Self::Arg5),
                "parm7" => Some(Self::Arg6),
                "parm8" => Some(Self::Arg7),
                "parm9" => Some(Self::Arg8),
                "parm10" => Some(Self::Arg9),
                "parm11" => Some(Self::Arg10),
                "parm12" => Some(Self::Arg11),
                "parm13" => Some(Self::Arg12),
                "parm14" => Some(Self::Arg13),
                "parm15" => Some(Self::Arg14),
                "parm16" => Some(Self::Arg15),
                "v_forward" => Some(Self::VForward),
                "v_forward_x" => Some(Self::VForwardX),
                "v_forward_y" => Some(Self::VForwardY),
                "v_forward_z" => Some(Self::VForwardZ),
                "v_up" => Some(Self::VUp),
                "v_up_x" => Some(Self::VUpX),
                "v_up_y" => Some(Self::VUpY),
                "v_up_z" => Some(Self::VUpZ),
                "v_right" => Some(Self::VRight),
                "v_right_x" => Some(Self::VRightX),
                "v_right_y" => Some(Self::VRightY),
                "v_right_z" => Some(Self::VRightZ),
                "trace_allsolid" => Some(Self::TraceAllSolid),
                "trace_startsolid" => Some(Self::TraceStartSolid),
                "trace_fraction" => Some(Self::TraceFraction),
                "trace_endpos" => Some(Self::TraceEndPos),
                "trace_endpos_x" => Some(Self::TraceEndPosX),
                "trace_endpos_y" => Some(Self::TraceEndPosY),
                "trace_endpos_z" => Some(Self::TraceEndPosZ),
                "trace_plane_normal" => Some(Self::TracePlaneNormal),
                "trace_plane_normal_x" => Some(Self::TracePlaneNormalX),
                "trace_plane_normal_y" => Some(Self::TracePlaneNormalY),
                "trace_plane_normal_z" => Some(Self::TracePlaneNormalZ),
                "trace_plane_dist" => Some(Self::TracePlaneDist),
                "trace_ent" => Some(Self::TraceEntity),
                "trace_inopen" => Some(Self::TraceInOpen),
                "trace_inwater" => Some(Self::TraceInWater),
                "msg_entity" => Some(Self::MsgEntity),
                "main" => Some(Self::Main),
                "StartFrame" => Some(Self::StartFrame),
                "PlayerPreThink" => Some(Self::PlayerPreThink),
                "PlayerPostThink" => Some(Self::PlayerPostThink),
                "ClientKill" => Some(Self::ClientKill),
                "ClientConnect" => Some(Self::ClientConnect),
                "PutClientInServer" => Some(Self::PutClientInServer),
                "ClientDisconnect" => Some(Self::ClientDisconnect),
                "SetNewParms" => Some(Self::SetNewArgs),
                "SetChangeParms" => Some(Self::SetChangeArgs),
                _ => None,
            }
        }

        pub const fn type_(&self) -> Type {
            match self {
                Self::Self_ => Type::Scalar(ScalarType::Entity),
                Self::Other => Type::Scalar(ScalarType::Entity),
                Self::World => Type::Scalar(ScalarType::Entity),
                Self::Time => Type::Scalar(ScalarType::Float),
                Self::FrameTime => Type::Scalar(ScalarType::Float),
                // Self::NewMissile => Type::Scalar(ScalarType::Entity),
                Self::ForceRetouch => Type::Scalar(ScalarType::Float),
                Self::MapName => Type::Scalar(ScalarType::String),
                Self::Deathmatch => Type::Scalar(ScalarType::Float),
                Self::Coop => Type::Scalar(ScalarType::Float),
                Self::TeamPlay => Type::Scalar(ScalarType::Float),
                Self::ServerFlags => Type::Scalar(ScalarType::Float),
                Self::TotalSecrets => Type::Scalar(ScalarType::Float),
                Self::TotalMonsters => Type::Scalar(ScalarType::Float),
                Self::FoundSecrets => Type::Scalar(ScalarType::Float),
                Self::KilledMonsters => Type::Scalar(ScalarType::Float),
                Self::Arg0 => Type::Scalar(ScalarType::Float),
                Self::Arg1 => Type::Scalar(ScalarType::Float),
                Self::Arg2 => Type::Scalar(ScalarType::Float),
                Self::Arg3 => Type::Scalar(ScalarType::Float),
                Self::Arg4 => Type::Scalar(ScalarType::Float),
                Self::Arg5 => Type::Scalar(ScalarType::Float),
                Self::Arg6 => Type::Scalar(ScalarType::Float),
                Self::Arg7 => Type::Scalar(ScalarType::Float),
                Self::Arg8 => Type::Scalar(ScalarType::Float),
                Self::Arg9 => Type::Scalar(ScalarType::Float),
                Self::Arg10 => Type::Scalar(ScalarType::Float),
                Self::Arg11 => Type::Scalar(ScalarType::Float),
                Self::Arg12 => Type::Scalar(ScalarType::Float),
                Self::Arg13 => Type::Scalar(ScalarType::Float),
                Self::Arg14 => Type::Scalar(ScalarType::Float),
                Self::Arg15 => Type::Scalar(ScalarType::Float),
                Self::VForward => Type::Vector,
                Self::VForwardX => Type::Scalar(ScalarType::Float),
                Self::VForwardY => Type::Scalar(ScalarType::Float),
                Self::VForwardZ => Type::Scalar(ScalarType::Float),
                Self::VUp => Type::Vector,
                Self::VUpX => Type::Scalar(ScalarType::Float),
                Self::VUpY => Type::Scalar(ScalarType::Float),
                Self::VUpZ => Type::Scalar(ScalarType::Float),
                Self::VRight => Type::Vector,
                Self::VRightX => Type::Scalar(ScalarType::Float),
                Self::VRightY => Type::Scalar(ScalarType::Float),
                Self::VRightZ => Type::Scalar(ScalarType::Float),
                Self::TraceAllSolid => Type::Scalar(ScalarType::Float),
                Self::TraceStartSolid => Type::Scalar(ScalarType::Float),
                Self::TraceFraction => Type::Scalar(ScalarType::Float),
                Self::TraceEndPos => Type::Vector,
                Self::TraceEndPosX => Type::Scalar(ScalarType::Float),
                Self::TraceEndPosY => Type::Scalar(ScalarType::Float),
                Self::TraceEndPosZ => Type::Scalar(ScalarType::Float),
                Self::TracePlaneNormal => Type::Vector,
                Self::TracePlaneNormalX => Type::Scalar(ScalarType::Float),
                Self::TracePlaneNormalY => Type::Scalar(ScalarType::Float),
                Self::TracePlaneNormalZ => Type::Scalar(ScalarType::Float),
                Self::TracePlaneDist => Type::Scalar(ScalarType::Float),
                Self::TraceEntity => Type::Scalar(ScalarType::Entity),
                Self::TraceInOpen => Type::Scalar(ScalarType::Float),
                Self::TraceInWater => Type::Scalar(ScalarType::Float),
                Self::MsgEntity => Type::Scalar(ScalarType::Entity),

                // Functions
                Self::Main => Type::Scalar(ScalarType::Function),
                Self::StartFrame => Type::Scalar(ScalarType::Function),
                Self::PlayerPreThink => Type::Scalar(ScalarType::Function),
                Self::PlayerPostThink => Type::Scalar(ScalarType::Function),
                Self::ClientKill => Type::Scalar(ScalarType::Function),
                Self::ClientConnect => Type::Scalar(ScalarType::Function),
                Self::PutClientInServer => Type::Scalar(ScalarType::Function),
                Self::ClientDisconnect => Type::Scalar(ScalarType::Function),
                Self::SetNewArgs => Type::Scalar(ScalarType::Function),
                Self::SetChangeArgs => Type::Scalar(ScalarType::Function),
            }
        }
    }

    #[cfg(test)]
    mod test {
        use strum::IntoEnumIterator;

        use crate::quake1::globals::GlobalAddr;

        #[test]
        fn check_to_from_parity() {
            for glob in <GlobalAddr as IntoEnumIterator>::iter() {
                let (idx, ty) = (glob.to_u16(), glob.type_());

                assert_eq!(GlobalAddr::from_u16(idx, ty), Some(glob));

                let name = glob.name();

                assert_eq!(GlobalAddr::from_name(name), Some(glob));
            }
        }
    }
}

pub mod fields {
    use num_derive::FromPrimitive;
    use std::fmt;

    #[derive(Copy, Clone, Debug, PartialEq, Eq, FromPrimitive)]
    pub enum FieldAddrFloat {
        ModelIndex = 0,
        AbsMinX = 1,
        AbsMinY = 2,
        AbsMinZ = 3,
        AbsMaxX = 4,
        AbsMaxY = 5,
        AbsMaxZ = 6,
        /// Used by mobile level geometry such as moving platforms.
        LocalTime = 7,
        /// Determines the movement behavior of an entity. The value must be a variant of `MoveKind`.
        MoveKind = 8,
        Solid = 9,
        OriginX = 10,
        OriginY = 11,
        OriginZ = 12,
        OldOriginX = 13,
        OldOriginY = 14,
        OldOriginZ = 15,
        VelocityX = 16,
        VelocityY = 17,
        VelocityZ = 18,
        AnglesX = 19,
        AnglesY = 20,
        AnglesZ = 21,
        AngularVelocityX = 22,
        AngularVelocityY = 23,
        AngularVelocityZ = 24,
        PunchAngleX = 25,
        PunchAngleY = 26,
        PunchAngleZ = 27,
        /// The index of the entity's animation frame.
        FrameId = 30,
        /// The index of the entity's skin.
        SkinId = 31,
        /// Effects flags applied to the entity. See `EntityEffects`.
        Effects = 32,
        /// Minimum extent in local coordinates, X-coordinate.
        MinsX = 33,
        /// Minimum extent in local coordinates, Y-coordinate.
        MinsY = 34,
        /// Minimum extent in local coordinates, Z-coordinate.
        MinsZ = 35,
        /// Maximum extent in local coordinates, X-coordinate.
        MaxsX = 36,
        /// Maximum extent in local coordinates, Y-coordinate.
        MaxsY = 37,
        /// Maximum extent in local coordinates, Z-coordinate.
        MaxsZ = 38,
        SizeX = 39,
        SizeY = 40,
        SizeZ = 41,
        /// The next server time at which the entity should run its think function.
        NextThink = 46,
        /// The entity's remaining health.
        Health = 48,
        /// The number of kills scored by the entity.
        Frags = 49,
        Weapon = 50,
        WeaponFrame = 52,
        /// The entity's remaining ammunition for its selected weapon.
        CurrentAmmo = 53,
        /// The entity's remaining shotgun shells.
        AmmoShells = 54,
        /// The entity's remaining shotgun shells.
        AmmoNails = 55,
        /// The entity's remaining rockets/grenades.
        AmmoRockets = 56,
        AmmoCells = 57,
        Items = 58,
        TakeDamage = 59,
        DeadFlag = 61,
        ViewOffsetX = 62,
        ViewOffsetY = 63,
        ViewOffsetZ = 64,
        Button0 = 65,
        Button1 = 66,
        Button2 = 67,
        Impulse = 68,
        FixAngle = 69,
        ViewAngleX = 70,
        ViewAngleY = 71,
        ViewAngleZ = 72,
        IdealPitch = 73,
        Flags = 76,
        Colormap = 77,
        Team = 78,
        MaxHealth = 79,
        TeleportTime = 80,
        ArmorStrength = 81,
        ArmorValue = 82,
        WaterLevel = 83,
        Contents = 84,
        IdealYaw = 85,
        YawSpeed = 86,
        SpawnFlags = 89,
        DmgTake = 92,
        DmgSave = 93,
        MoveDirectionX = 96,
        MoveDirectionY = 97,
        MoveDirectionZ = 98,
        Sounds = 100,
    }

    impl fmt::Display for FieldAddrFloat {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ModelIndex => write!(f, "model_id"),
                Self::AbsMinX => write!(f, "abs_min_x"),
                Self::AbsMinY => write!(f, "abs_min_y"),
                Self::AbsMinZ => write!(f, "abs_min_z"),
                Self::AbsMaxX => write!(f, "abs_max_x"),
                Self::AbsMaxY => write!(f, "abs_max_y"),
                Self::AbsMaxZ => write!(f, "abs_max_z"),
                Self::LocalTime => write!(f, "local_time"),
                Self::MoveKind => write!(f, "move_kind"),
                Self::Solid => write!(f, "solid"),
                Self::OriginX => write!(f, "origin_x"),
                Self::OriginY => write!(f, "origin_y"),
                Self::OriginZ => write!(f, "origin_z"),
                Self::OldOriginX => write!(f, "old_origin_x"),
                Self::OldOriginY => write!(f, "old_origin_y"),
                Self::OldOriginZ => write!(f, "old_origin_z"),
                Self::VelocityX => write!(f, "velocity_x"),
                Self::VelocityY => write!(f, "velocity_y"),
                Self::VelocityZ => write!(f, "velocity_z"),
                Self::AnglesX => write!(f, "angles_x"),
                Self::AnglesY => write!(f, "angles_y"),
                Self::AnglesZ => write!(f, "angles_z"),
                Self::AngularVelocityX => write!(f, "angular_velocity_x"),
                Self::AngularVelocityY => write!(f, "angular_velocity_y"),
                Self::AngularVelocityZ => write!(f, "angular_velocity_z"),
                Self::PunchAngleX => write!(f, "punch_angle_x"),
                Self::PunchAngleY => write!(f, "punch_angle_y"),
                Self::PunchAngleZ => write!(f, "punch_angle_z"),
                Self::FrameId => write!(f, "frame_id"),
                Self::SkinId => write!(f, "skin_id"),
                Self::Effects => write!(f, "effects"),
                Self::MinsX => write!(f, "mins_x"),
                Self::MinsY => write!(f, "mins_y"),
                Self::MinsZ => write!(f, "mins_z"),
                Self::MaxsX => write!(f, "maxs_x"),
                Self::MaxsY => write!(f, "maxs_y"),
                Self::MaxsZ => write!(f, "maxs_z"),
                Self::SizeX => write!(f, "size_x"),
                Self::SizeY => write!(f, "size_y"),
                Self::SizeZ => write!(f, "size_z"),
                Self::NextThink => write!(f, "next_think"),
                Self::Health => write!(f, "health"),
                Self::Frags => write!(f, "frags"),
                Self::Weapon => write!(f, "weapon"),
                Self::WeaponFrame => write!(f, "weapon_frame"),
                Self::CurrentAmmo => write!(f, "current_ammo"),
                Self::AmmoShells => write!(f, "ammo_shells"),
                Self::AmmoNails => write!(f, "ammo_nails"),
                Self::AmmoRockets => write!(f, "ammo_rockets"),
                Self::AmmoCells => write!(f, "ammo_cells"),
                Self::Items => write!(f, "items"),
                Self::TakeDamage => write!(f, "take_damage"),
                Self::DeadFlag => write!(f, "dead_flag"),
                Self::ViewOffsetX => write!(f, "view_offset_x"),
                Self::ViewOffsetY => write!(f, "view_offset_y"),
                Self::ViewOffsetZ => write!(f, "view_offset_z"),
                Self::Button0 => write!(f, "button0"),
                Self::Button1 => write!(f, "button1"),
                Self::Button2 => write!(f, "button2"),
                Self::Impulse => write!(f, "impulse"),
                Self::FixAngle => write!(f, "fix_angle"),
                Self::ViewAngleX => write!(f, "view_angle_x"),
                Self::ViewAngleY => write!(f, "view_angle_y"),
                Self::ViewAngleZ => write!(f, "view_angle_z"),
                Self::IdealPitch => write!(f, "ideal_pitch"),
                Self::Flags => write!(f, "flags"),
                Self::Colormap => write!(f, "colormap"),
                Self::Team => write!(f, "team"),
                Self::MaxHealth => write!(f, "max_health"),
                Self::TeleportTime => write!(f, "teleport_time"),
                Self::ArmorStrength => write!(f, "armor_strength"),
                Self::ArmorValue => write!(f, "armor_value"),
                Self::WaterLevel => write!(f, "water_level"),
                Self::Contents => write!(f, "contents"),
                Self::IdealYaw => write!(f, "ideal_yaw"),
                Self::YawSpeed => write!(f, "yaw_speed"),
                Self::SpawnFlags => write!(f, "spawn_flags"),
                Self::DmgTake => write!(f, "dmg_take"),
                Self::DmgSave => write!(f, "dmg_save"),
                Self::MoveDirectionX => write!(f, "move_direction_x"),
                Self::MoveDirectionY => write!(f, "move_direction_y"),
                Self::MoveDirectionZ => write!(f, "move_direction_z"),
                Self::Sounds => write!(f, "sounds"),
            }
        }
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, FromPrimitive)]
    pub enum FieldAddrVector {
        AbsMin = 1,
        AbsMax = 4,
        Origin = 10,
        OldOrigin = 13,
        Velocity = 16,
        Angles = 19,
        AngularVelocity = 22,
        PunchAngle = 25,
        Mins = 33,
        Maxs = 36,
        Size = 39,
        ViewOffset = 62,
        ViewAngle = 70,
        MoveDirection = 96,
    }

    #[derive(Copy, Clone, Debug, FromPrimitive)]
    pub enum FieldAddrStringId {
        ClassName = 28,
        ModelName = 29,
        WeaponModelName = 51,
        NetName = 74,
        Target = 90,
        TargetName = 91,
        Message = 99,
        Noise0Name = 101,
        Noise1Name = 102,
        Noise2Name = 103,
        Noise3Name = 104,
    }

    impl fmt::Display for FieldAddrStringId {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::ClassName => write!(f, "classname"),
                Self::ModelName => write!(f, "modelname"),
                Self::WeaponModelName => write!(f, "weaponmodelname"),
                Self::NetName => write!(f, "netname"),
                Self::Target => write!(f, "target"),
                Self::TargetName => write!(f, "targetname"),
                Self::Message => write!(f, "message"),
                Self::Noise0Name => write!(f, "noise0name"),
                Self::Noise1Name => write!(f, "noise1name"),
                Self::Noise2Name => write!(f, "noise2name"),
                Self::Noise3Name => write!(f, "noise3name"),
            }
        }
    }

    #[derive(Copy, Clone, Debug, FromPrimitive)]
    pub enum FieldAddrEntityId {
        /// The entity this entity is standing on.
        Ground = 47,
        Chain = 60,
        Enemy = 75,
        Aim = 87,
        Goal = 88,
        DmgInflictor = 94,
        Owner = 95,
    }

    #[derive(Copy, Clone, Debug, FromPrimitive)]
    pub enum FieldAddrFunctionId {
        Touch = 42,
        Use = 43,
        Think = 44,
        Blocked = 45,
    }
}
