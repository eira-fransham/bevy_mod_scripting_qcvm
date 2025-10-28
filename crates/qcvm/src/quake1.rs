//! # Quake 1 field, global and builtin definitions
//!
//! This module contains Quake-compatible builtin IDs, which should be implemented
//! if writing a host that can load unmodified `progs.dat` files designed for the
//! vanilla Quake engine.
//!
//! This is compatible with [`progdefs.q1`](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).

pub use globals::GLOBALS_RANGE;

pub mod globals {
    //! Global definitions for `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).

    use std::ops::Range;

    use strum::EnumIter;

    use crate::Type;

    /// The first static global address.
    pub const GLOBALS_START: u32 = 28;

    /// Seismon has "true" locals and does not reuse globals for locals, for correctness and
    /// resiliency.
    pub const LOCALS_START: u32 = GlobalAddr::SetChangeArgs.to_u16() as u32 + 1;

    /// The range of static global addresses.
    pub const GLOBALS_RANGE: Range<usize> = GLOBALS_START as usize..LOCALS_START as usize;

    /// Global indices for globals defined in `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).
    #[derive(EnumIter, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum GlobalAddr {
        /// entity self
        Self_,
        /// entity other
        Other,
        /// entity world
        World,
        /// float time
        Time,
        /// float frametime
        FrameTime,
        /// float force_retouch
        ForceRetouch,
        /// string mapname
        MapName,
        /// float deathmatch
        Deathmatch,
        /// float coop
        Coop,
        /// float teamplay
        TeamPlay,
        /// float serverflags
        ServerFlags,
        /// float total_secrets
        TotalSecrets,
        /// float total_monsters
        TotalMonsters,
        /// float found_secrets
        FoundSecrets,
        /// float killed_monsters
        KilledMonsters,
        /// float parm1
        Arg0,
        /// float parm2
        Arg1,
        /// float parm3
        Arg2,
        /// float parm4
        Arg3,
        /// float parm5
        Arg4,
        /// float parm6
        Arg5,
        /// float parm7
        Arg6,
        /// float parm8
        Arg7,
        /// float parm9
        Arg8,
        /// float parm10
        Arg9,
        /// float parm11
        Arg10,
        /// float parm12
        Arg11,
        /// float parm13
        Arg12,
        /// float parm14
        Arg13,
        /// float parm15
        Arg14,
        /// float parm16
        Arg15,
        /// vector v_forward
        VForward,
        /// float v_forward_x
        VForwardX,
        /// float v_forward_y
        VForwardY,
        /// float v_forward_z
        VForwardZ,
        /// vector v_up
        VUp,
        /// float v_up_x
        VUpX,
        /// float v_up_y
        VUpY,
        /// float v_up_z
        VUpZ,
        /// vector v_right
        VRight,
        /// float v_right_x
        VRightX,
        /// float v_right_y
        VRightY,
        /// float v_right_z
        VRightZ,
        /// float trace_allsolid
        TraceAllSolid,
        /// float trace_startsolid
        TraceStartSolid,
        /// float trace_fraction
        TraceFraction,
        /// vector trace_endpos
        TraceEndPos,
        /// float trace_endpos_x
        TraceEndPosX,
        /// float trace_endpos_y
        TraceEndPosY,
        /// float trace_endpos_z
        TraceEndPosZ,
        /// vector trace_plane_normal
        TracePlaneNormal,
        /// float trace_plane_normal_x
        TracePlaneNormalX,
        /// float trace_plane_normal_y
        TracePlaneNormalY,
        /// float trace_plane_normal_z
        TracePlaneNormalZ,
        /// float trace_plane_dist
        TracePlaneDist,
        /// entity trace_ent
        TraceEntity,
        /// float trace_inopen
        TraceInOpen,
        /// float trace_inwater
        TraceInWater,
        /// entity msg_entity
        MsgEntity,

        // -------------- Functions --------------
        /// function main
        Main,
        /// function StartFrame
        StartFrame,
        /// function PlayerPreThink
        PlayerPreThink,
        /// function PlayerPostThink
        PlayerPostThink,
        /// function ClientKill
        ClientKill,
        /// function ClientConnect
        ClientConnect,
        /// function PutClientInServer
        PutClientInServer,
        /// function ClientDisconnect
        ClientDisconnect,
        /// function SetNewParms
        SetNewArgs,
        /// function SetChangeParms
        SetChangeArgs,
    }

    impl GlobalAddr {
        /// Convert a raw offset into the relevant `GlobalAddr`, given the type. Certain
        /// field offsets are specified to overlap in the `progdefs.qc` in order to have
        /// quick access to fields of vectors, and this can distinguish between `vector foo`
        /// and `float foo_x`.
        pub const fn from_u16(val: u16, ty: Type) -> Option<Self> {
            match (val, ty) {
                (28, Type::Entity) => Some(Self::Self_),
                (29, Type::Entity) => Some(Self::Other),
                (30, Type::Entity) => Some(Self::World),
                (31, Type::Float) => Some(Self::Time),
                (32, Type::Float) => Some(Self::FrameTime),
                (33, Type::Float) => Some(Self::ForceRetouch),
                (34, Type::String) => Some(Self::MapName),
                (35, Type::Float) => Some(Self::Deathmatch),
                (36, Type::Float) => Some(Self::Coop),
                (37, Type::Float) => Some(Self::TeamPlay),
                (38, Type::Float) => Some(Self::ServerFlags),
                (39, Type::Float) => Some(Self::TotalSecrets),
                (40, Type::Float) => Some(Self::TotalMonsters),
                (41, Type::Float) => Some(Self::FoundSecrets),
                (42, Type::Float) => Some(Self::KilledMonsters),
                (43, Type::Float) => Some(Self::Arg0),
                (44, Type::Float) => Some(Self::Arg1),
                (45, Type::Float) => Some(Self::Arg2),
                (46, Type::Float) => Some(Self::Arg3),
                (47, Type::Float) => Some(Self::Arg4),
                (48, Type::Float) => Some(Self::Arg5),
                (49, Type::Float) => Some(Self::Arg6),
                (50, Type::Float) => Some(Self::Arg7),
                (51, Type::Float) => Some(Self::Arg8),
                (52, Type::Float) => Some(Self::Arg9),
                (53, Type::Float) => Some(Self::Arg10),
                (54, Type::Float) => Some(Self::Arg11),
                (55, Type::Float) => Some(Self::Arg12),
                (56, Type::Float) => Some(Self::Arg13),
                (57, Type::Float) => Some(Self::Arg14),
                (58, Type::Float) => Some(Self::Arg15),
                (59, Type::Vector) => Some(Self::VForward),
                (59, Type::Float) => Some(Self::VForwardX),
                (60, Type::Float) => Some(Self::VForwardY),
                (61, Type::Float) => Some(Self::VForwardZ),
                (62, Type::Vector) => Some(Self::VUp),
                (62, Type::Float) => Some(Self::VUpX),
                (63, Type::Float) => Some(Self::VUpY),
                (64, Type::Float) => Some(Self::VUpZ),
                (65, Type::Vector) => Some(Self::VRight),
                (65, Type::Float) => Some(Self::VRightX),
                (66, Type::Float) => Some(Self::VRightY),
                (67, Type::Float) => Some(Self::VRightZ),
                (68, Type::Float) => Some(Self::TraceAllSolid),
                (69, Type::Float) => Some(Self::TraceStartSolid),
                (70, Type::Float) => Some(Self::TraceFraction),
                (71, Type::Vector) => Some(Self::TraceEndPos),
                (71, Type::Float) => Some(Self::TraceEndPosX),
                (72, Type::Float) => Some(Self::TraceEndPosY),
                (73, Type::Float) => Some(Self::TraceEndPosZ),
                (74, Type::Vector) => Some(Self::TracePlaneNormal),
                (74, Type::Float) => Some(Self::TracePlaneNormalX),
                (75, Type::Float) => Some(Self::TracePlaneNormalY),
                (76, Type::Float) => Some(Self::TracePlaneNormalZ),
                (77, Type::Float) => Some(Self::TracePlaneDist),
                (78, Type::Entity) => Some(Self::TraceEntity),
                (79, Type::Float) => Some(Self::TraceInOpen),
                (80, Type::Float) => Some(Self::TraceInWater),
                (81, Type::Entity) => Some(Self::MsgEntity),

                // Functions
                (82, Type::Function) => Some(Self::Main),
                (83, Type::Function) => Some(Self::StartFrame),
                (84, Type::Function) => Some(Self::PlayerPreThink),
                (85, Type::Function) => Some(Self::PlayerPostThink),
                (86, Type::Function) => Some(Self::ClientKill),
                (87, Type::Function) => Some(Self::ClientConnect),
                (88, Type::Function) => Some(Self::PutClientInServer),
                (89, Type::Function) => Some(Self::ClientDisconnect),
                (90, Type::Function) => Some(Self::SetNewArgs),
                (91, Type::Function) => Some(Self::SetChangeArgs),
                _ => None,
            }
        }

        /// Get the offset to this global.
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

        /// Get the name of this global.
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

        /// Given a name, get the global address the name corresponds to.
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

        /// Get the type of this global.
        pub const fn type_(&self) -> Type {
            match self {
                Self::Self_ => Type::Entity,
                Self::Other => Type::Entity,
                Self::World => Type::Entity,
                Self::Time => Type::Float,
                Self::FrameTime => Type::Float,
                // Self::NewMissile => Type::Scalar(ScalarType::Entity),
                Self::ForceRetouch => Type::Float,
                Self::MapName => Type::String,
                Self::Deathmatch => Type::Float,
                Self::Coop => Type::Float,
                Self::TeamPlay => Type::Float,
                Self::ServerFlags => Type::Float,
                Self::TotalSecrets => Type::Float,
                Self::TotalMonsters => Type::Float,
                Self::FoundSecrets => Type::Float,
                Self::KilledMonsters => Type::Float,
                Self::Arg0 => Type::Float,
                Self::Arg1 => Type::Float,
                Self::Arg2 => Type::Float,
                Self::Arg3 => Type::Float,
                Self::Arg4 => Type::Float,
                Self::Arg5 => Type::Float,
                Self::Arg6 => Type::Float,
                Self::Arg7 => Type::Float,
                Self::Arg8 => Type::Float,
                Self::Arg9 => Type::Float,
                Self::Arg10 => Type::Float,
                Self::Arg11 => Type::Float,
                Self::Arg12 => Type::Float,
                Self::Arg13 => Type::Float,
                Self::Arg14 => Type::Float,
                Self::Arg15 => Type::Float,
                Self::VForward => Type::Vector,
                Self::VForwardX => Type::Float,
                Self::VForwardY => Type::Float,
                Self::VForwardZ => Type::Float,
                Self::VUp => Type::Vector,
                Self::VUpX => Type::Float,
                Self::VUpY => Type::Float,
                Self::VUpZ => Type::Float,
                Self::VRight => Type::Vector,
                Self::VRightX => Type::Float,
                Self::VRightY => Type::Float,
                Self::VRightZ => Type::Float,
                Self::TraceAllSolid => Type::Float,
                Self::TraceStartSolid => Type::Float,
                Self::TraceFraction => Type::Float,
                Self::TraceEndPos => Type::Vector,
                Self::TraceEndPosX => Type::Float,
                Self::TraceEndPosY => Type::Float,
                Self::TraceEndPosZ => Type::Float,
                Self::TracePlaneNormal => Type::Vector,
                Self::TracePlaneNormalX => Type::Float,
                Self::TracePlaneNormalY => Type::Float,
                Self::TracePlaneNormalZ => Type::Float,
                Self::TracePlaneDist => Type::Float,
                Self::TraceEntity => Type::Entity,
                Self::TraceInOpen => Type::Float,
                Self::TraceInWater => Type::Float,
                Self::MsgEntity => Type::Entity,

                // Functions
                Self::Main => Type::Function,
                Self::StartFrame => Type::Function,
                Self::PlayerPreThink => Type::Function,
                Self::PlayerPostThink => Type::Function,
                Self::ClientKill => Type::Function,
                Self::ClientConnect => Type::Function,
                Self::PutClientInServer => Type::Function,
                Self::ClientDisconnect => Type::Function,
                Self::SetNewArgs => Type::Function,
                Self::SetChangeArgs => Type::Function,
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

// TODO
#[doc(hidden)]
pub mod fields {
    //! Field definitions for `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).

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
