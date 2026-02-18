//! # Quake 1 field, global and builtin definitions
//! const MAP
//!
//! This module contains Quake-compatible builtin IDs, which should be implemented
//! if writing a host that can load unmodified `progs.dat` files designed for the
//! vanilla Quake engine.
//!
//! This is compatible with [`progdefs.q1`](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).

pub use self::{fields::FieldAddr, globals::GlobalAddr};

/// Global definitions for `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).
pub mod globals {
    use std::fmt;

    use num::FromPrimitive;
    use strum::{EnumIter, VariantArray};

    use crate::{Address, Type, VectorField};

    // int      pad[28];
    // int      self;
    // int      other;
    // int      world;
    // float    time;
    // float    frametime;
    // float    force_retouch;
    // string_t mapname;
    // float    deathmatch;
    // float    coop;
    // float    teamplay;
    // float    serverflags;
    // float    total_secrets;
    // float    total_monsters;
    // float    found_secrets;
    // float    killed_monsters;
    // float    parm1;
    // float    parm2;
    // float    parm3;
    // float    parm4;
    // float    parm5;
    // float    parm6;
    // float    parm7;
    // float    parm8;
    // float    parm9;
    // float    parm10;
    // float    parm11;
    // float    parm12;
    // float    parm13;
    // float    parm14;
    // float    parm15;
    // float    parm16;
    // vec3_t   v_forward;
    // vec3_t   v_up;
    // vec3_t   v_right;
    // float    trace_allsolid;
    // float    trace_startsolid;
    // float    trace_fraction;
    // vec3_t   trace_endpos;
    // vec3_t   trace_plane_normal;
    // float    trace_plane_dist;
    // int      trace_ent;
    // float    trace_inopen;
    // float    trace_inwater;
    // int      msg_entity;
    // func_t   main;
    // func_t   StartFrame;
    // func_t   PlayerPreThink;
    // func_t   PlayerPostThink;
    // func_t   ClientKill;
    // func_t   ClientConnect;
    // func_t   PutClientInServer;
    // func_t   ClientDisconnect;
    // func_t   SetNewParms;
    // func_t   SetChangeParms;

    /// Global indices for globals defined in `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).
    #[derive(VariantArray, EnumIter, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        SpawnParam0,
        /// float parm2
        SpawnParam1,
        /// float parm3
        SpawnParam2,
        /// float parm4
        SpawnParam3,
        /// float parm5
        SpawnParam4,
        /// float parm6
        SpawnParam5,
        /// float parm7
        SpawnParam6,
        /// float parm8
        SpawnParam7,
        /// float parm9
        SpawnParam8,
        /// float parm10
        SpawnParam9,
        /// float parm11
        SpawnParam10,
        /// float parm12
        SpawnParam11,
        /// float parm13
        SpawnParam12,
        /// float parm14
        SpawnParam13,
        /// float parm15
        SpawnParam14,
        /// float parm16
        SpawnParam15,
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

    impl fmt::Display for GlobalAddr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.name())
        }
    }

    impl FromPrimitive for GlobalAddr {
        fn from_u16(n: u16) -> Option<Self> {
            Self::from_u16_typed(n, Type::AnyScalar)
        }

        fn from_i64(n: i64) -> Option<Self> {
            FromPrimitive::from_u16(n.try_into().ok()?)
        }

        fn from_u64(n: u64) -> Option<Self> {
            FromPrimitive::from_u16(n.try_into().ok()?)
        }
    }

    impl Address for GlobalAddr {
        fn vector_field_or_scalar(&self) -> (Self, VectorField) {
            match self {
                GlobalAddr::VForwardX => (GlobalAddr::VForward, VectorField::XOrScalar),
                GlobalAddr::VForwardY => (GlobalAddr::VForward, VectorField::Y),
                GlobalAddr::VForwardZ => (GlobalAddr::VForward, VectorField::Z),
                GlobalAddr::VUpX => (GlobalAddr::VUp, VectorField::XOrScalar),
                GlobalAddr::VUpY => (GlobalAddr::VUp, VectorField::Y),
                GlobalAddr::VUpZ => (GlobalAddr::VUp, VectorField::Z),
                GlobalAddr::VRightX => (GlobalAddr::VRight, VectorField::XOrScalar),
                GlobalAddr::VRightY => (GlobalAddr::VRight, VectorField::Y),
                GlobalAddr::VRightZ => (GlobalAddr::VRight, VectorField::Z),
                GlobalAddr::TraceEndPosX => (GlobalAddr::TraceEndPos, VectorField::XOrScalar),
                GlobalAddr::TraceEndPosY => (GlobalAddr::TraceEndPos, VectorField::Y),
                GlobalAddr::TraceEndPosZ => (GlobalAddr::TraceEndPos, VectorField::Z),
                GlobalAddr::TracePlaneNormalX => {
                    (GlobalAddr::TracePlaneNormal, VectorField::XOrScalar)
                }
                GlobalAddr::TracePlaneNormalY => (GlobalAddr::TracePlaneNormal, VectorField::Y),
                GlobalAddr::TracePlaneNormalZ => (GlobalAddr::TracePlaneNormal, VectorField::Z),
                other => (*other, VectorField::XOrScalar),
            }
        }

        fn to_u16(&self) -> u16 {
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
                Self::SpawnParam0 => 43,
                Self::SpawnParam1 => 44,
                Self::SpawnParam2 => 45,
                Self::SpawnParam3 => 46,
                Self::SpawnParam4 => 47,
                Self::SpawnParam5 => 48,
                Self::SpawnParam6 => 49,
                Self::SpawnParam7 => 50,
                Self::SpawnParam8 => 51,
                Self::SpawnParam9 => 52,
                Self::SpawnParam10 => 53,
                Self::SpawnParam11 => 54,
                Self::SpawnParam12 => 55,
                Self::SpawnParam13 => 56,
                Self::SpawnParam14 => 57,
                Self::SpawnParam15 => 58,
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

        fn name(&self) -> &'static str {
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
                Self::SpawnParam0 => "parm1",
                Self::SpawnParam1 => "parm2",
                Self::SpawnParam2 => "parm3",
                Self::SpawnParam3 => "parm4",
                Self::SpawnParam4 => "parm5",
                Self::SpawnParam5 => "parm6",
                Self::SpawnParam6 => "parm7",
                Self::SpawnParam7 => "parm8",
                Self::SpawnParam8 => "parm9",
                Self::SpawnParam9 => "parm10",
                Self::SpawnParam10 => "parm11",
                Self::SpawnParam11 => "parm12",
                Self::SpawnParam12 => "parm13",
                Self::SpawnParam13 => "parm14",
                Self::SpawnParam14 => "parm15",
                Self::SpawnParam15 => "parm16",
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

        fn from_name(name: &str) -> Option<Self> {
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
                "parm1" => Some(Self::SpawnParam0),
                "parm2" => Some(Self::SpawnParam1),
                "parm3" => Some(Self::SpawnParam2),
                "parm4" => Some(Self::SpawnParam3),
                "parm5" => Some(Self::SpawnParam4),
                "parm6" => Some(Self::SpawnParam5),
                "parm7" => Some(Self::SpawnParam6),
                "parm8" => Some(Self::SpawnParam7),
                "parm9" => Some(Self::SpawnParam8),
                "parm10" => Some(Self::SpawnParam9),
                "parm11" => Some(Self::SpawnParam10),
                "parm12" => Some(Self::SpawnParam11),
                "parm13" => Some(Self::SpawnParam12),
                "parm14" => Some(Self::SpawnParam13),
                "parm15" => Some(Self::SpawnParam14),
                "parm16" => Some(Self::SpawnParam15),
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

        fn type_(&self) -> Type {
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
                Self::SpawnParam0 => Type::Float,
                Self::SpawnParam1 => Type::Float,
                Self::SpawnParam2 => Type::Float,
                Self::SpawnParam3 => Type::Float,
                Self::SpawnParam4 => Type::Float,
                Self::SpawnParam5 => Type::Float,
                Self::SpawnParam6 => Type::Float,
                Self::SpawnParam7 => Type::Float,
                Self::SpawnParam8 => Type::Float,
                Self::SpawnParam9 => Type::Float,
                Self::SpawnParam10 => Type::Float,
                Self::SpawnParam11 => Type::Float,
                Self::SpawnParam12 => Type::Float,
                Self::SpawnParam13 => Type::Float,
                Self::SpawnParam14 => Type::Float,
                Self::SpawnParam15 => Type::Float,
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

        use crate::{Address as _, quake1::globals::GlobalAddr};

        #[test]
        fn check_to_from_parity() {
            for glob in <GlobalAddr as IntoEnumIterator>::iter() {
                let (idx, ty) = (glob.to_u16(), glob.type_());

                assert_eq!(GlobalAddr::from_u16_typed(idx, ty), Some(glob));

                let name = glob.name();

                assert_eq!(GlobalAddr::from_name(name), Some(glob));
            }
        }
    }
}

/// Field definitions for `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/master/WinQuake/progdefs.q1).
pub mod fields {
    use num::FromPrimitive;
    use std::fmt;
    use strum::{EnumIter, VariantArray};

    use crate::{Address, Type, VectorField};

    // float    modelindex;
    // vec3_t   absmin;
    // vec3_t   absmax;
    // float    ltime;
    // float    movetype;
    // float    solid;
    // vec3_t   origin;
    // vec3_t   oldorigin;
    // vec3_t   velocity;
    // vec3_t   angles;
    // vec3_t   avelocity;
    // vec3_t   punchangle;
    // string_t classname;
    // string_t model;
    // float    frame;
    // float    skin;
    // float    effects;
    // vec3_t   mins;
    // vec3_t   maxs;
    // vec3_t   size;
    // func_t   touch;
    // func_t   use;
    // func_t   think;
    // func_t   blocked;
    // float    nextthink;
    // int      groundentity;
    // float    health;
    // float    frags;
    // float    weapon;
    // string_t weaponmodel;
    // float    weaponframe;
    // float    currentammo;
    // float    ammo_shells;
    // float    ammo_nails;
    // float    ammo_rockets;
    // float    ammo_cells;
    // float    items;
    // float    takedamage;
    // int      chain;
    // float    deadflag;
    // vec3_t   view_ofs;
    // float    button0;
    // float    button1;
    // float    button2;
    // float    impulse;
    // float    fixangle;
    // vec3_t   v_angle;
    // float    idealpitch;
    // string_t netname;
    // int      enemy;
    // float    flags;
    // float    colormap;
    // float    team;
    // float    max_health;
    // float    teleport_time;
    // float    armortype;
    // float    armorvalue;
    // float    waterlevel;
    // float    watertype;
    // float    ideal_yaw;
    // float    yaw_speed;
    // int      aiment;
    // int      goalentity;
    // float    spawnflags;
    // string_t target;
    // string_t targetname;
    // float    dmg_take;
    // float    dmg_save;
    // int      dmg_inflictor;
    // int      owner;
    // vec3_t   movedir;
    // string_t message;
    // float    sounds;
    // string_t noise;
    // string_t noise1;
    // string_t noise2;
    // string_t noise3;

    /// Indices for entity fields defined in `progdefs.q1`, see [the Quake GPL release](https://github.com/id-Software/Quake/blob/bf4ac424ce754894ac8f1dae6a3981954bc9852d/WinQuake/progdefs.q1#L62-L141).
    #[derive(VariantArray, EnumIter, Copy, Clone, Debug, PartialEq, Eq)]
    pub enum FieldAddr {
        /// float    modelindex;
        ModelId,
        /// vec3_t   absmin;
        AbsMin,
        /// float absmin.x;
        AbsMinX,
        /// float absmin.y;
        AbsMinY,
        /// float absmin.z;
        AbsMinZ,
        /// vec3_t   absmax;
        AbsMax,
        /// float absmax.x;
        AbsMaxX,
        /// float absmax.y;
        AbsMaxY,
        /// float absmax.z;
        AbsMaxZ,
        /// float    ltime;
        LocalTime,
        /// float    movetype;
        MoveType,
        /// float    solid;
        Solid,
        /// vec3_t   origin;
        Origin,
        /// float origin.x;
        OriginX,
        /// float origin.y;
        OriginY,
        /// float origin.z;
        OriginZ,
        /// vec3_t   oldorigin;
        OldOrigin,
        /// float oldorigin.x;
        OldOriginX,
        /// float oldorigin.y;
        OldOriginY,
        /// float oldorigin.z;
        OldOriginZ,
        /// vec3_t   velocity;
        Velocity,
        /// float velocity.x;
        VelocityX,
        /// float velocity.y;
        VelocityY,
        /// float velocity.z;
        VelocityZ,
        /// vec3_t   angles;
        Angles,
        /// float angles.x;
        AnglesX,
        /// float angles.y;
        AnglesY,
        /// float angles.z;
        AnglesZ,
        /// vec3_t   avelocity;
        AngularVelocity,
        /// float avelocity.x;
        AngularVelocityX,
        /// float avelocity.y;
        AngularVelocityY,
        /// float avelocity.z;
        AngularVelocityZ,
        /// vec3_t   punchangle;
        PunchAngle,
        /// float punchangle.x;
        PunchAngleX,
        /// float punchangle.y;
        PunchAngleY,
        /// float punchangle.z;
        PunchAngleZ,
        /// string_t classname;
        ClassName,
        /// string_t model;
        ModelName,
        /// float    frame;
        Frame,
        /// float    skin;
        SkinId,
        /// float    effects;
        Effects,
        /// vec3_t   mins;
        Mins,
        /// float mins.x;
        MinsX,
        /// float mins.y;
        MinsY,
        /// float mins.z;
        MinsZ,
        /// vec3_t   maxs;
        Maxs,
        /// float maxs.x;
        MaxsX,
        /// float maxs.y;
        MaxsY,
        /// float maxs.z;
        MaxsZ,
        /// vec3_t   size;
        Size,
        /// float size.x;
        SizeX,
        /// float size.y;
        SizeY,
        /// float size.z;
        SizeZ,
        /// func_t   touch;
        OnTouch,
        /// func_t   use;
        OnUse,
        /// func_t   think;
        OnThink,
        /// func_t   blocked;
        OnBlocked,
        /// float    nextthink;
        NextThink,
        /// int      groundentity;
        Ground,
        /// float    health;
        Health,
        /// float    frags;
        Frags,
        /// float    weapon;
        WeaponId,
        /// string_t weaponmodel;
        WeaponModelName,
        /// float    weaponframe;
        WeaponFrame,
        /// float    currentammo;
        CurrentAmmo,
        /// float    ammo_shells;
        AmmoShells,
        /// float    ammo_nails;
        AmmoNails,
        /// float    ammo_rockets;
        AmmoRockets,
        /// float    ammo_cells;
        AmmoCells,
        /// float    items;
        Items,
        /// float    takedamage;
        TakeDamage,
        /// int      chain;
        Chain,
        /// float    deadflag;
        DeadFlag,
        /// vec3_t   view_ofs;
        ViewOffset,
        /// float view_ofs.x;
        ViewOffsetX,
        /// float view_ofs.y;
        ViewOffsetY,
        /// float view_ofs.z;
        ViewOffsetZ,
        /// float    button0;
        Button0,
        /// float    button1;
        Button1,
        /// float    button2;
        Button2,
        /// float    impulse;
        Impulse,
        /// float    fixangle;
        FixAngle,
        /// vec3_t   v_angle;
        ViewAngle,
        /// float v_angle.x;
        ViewAngleX,
        /// float v_angle.y;
        ViewAngleY,
        /// float v_angle.z;
        ViewAngleZ,
        /// float    idealpitch;
        IdealPitch,
        /// string_t netname;
        NetworkName,
        /// int      enemy;
        Enemy,
        /// float    flags;
        Flags,
        /// float    colormap;
        ColorMapId,
        /// float    team;
        Team,
        /// float    max_health;
        MaxHealth,
        /// float    teleport_time;
        TeleportTime,
        /// float    armortype;
        ArmorType,
        /// float    armorvalue;
        ArmorValue,
        /// float    waterlevel;
        WaterLevel,
        /// float    watertype;
        WaterType,
        /// float    ideal_yaw;
        IdealYaw,
        /// float    yaw_speed;
        YawSpeed,
        /// int      aiment;
        AimEntity,
        /// int      goalentity;
        GoalEntity,
        /// float    spawnflags;
        SpawnFlags,
        /// string_t target;
        Target,
        /// string_t targetname;
        TargetName,
        /// float    dmg_take;
        DamageTaken,
        /// float    dmg_save;
        DamageSaved,
        /// int      dmg_inflictor;
        DamageInflictor,
        /// int      owner;
        Owner,
        /// vec3_t   movedir;
        MoveDirection,
        /// float movedir.x;
        MoveDirectionX,
        /// float movedir.y;
        MoveDirectionY,
        /// float movedir.z;
        MoveDirectionZ,
        /// string_t message;
        Message,
        /// float    sounds;
        Sounds,
        /// string_t noise;
        Noise0,
        /// string_t noise1;
        Noise1,
        /// string_t noise2;
        Noise2,
        /// string_t noise3;
        Noise3,
    }

    impl fmt::Display for FieldAddr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.name())
        }
    }

    impl FromPrimitive for FieldAddr {
        fn from_u16(n: u16) -> Option<Self> {
            Self::from_u16_typed(n, Type::AnyScalar)
        }

        fn from_i64(n: i64) -> Option<Self> {
            FromPrimitive::from_u16(n.try_into().ok()?)
        }

        fn from_u64(n: u64) -> Option<Self> {
            FromPrimitive::from_u16(n.try_into().ok()?)
        }
    }

    impl Address for FieldAddr {
        fn vector_field_or_scalar(&self) -> (Self, VectorField) {
            match self {
                Self::AbsMinX => (Self::AbsMin, VectorField::XOrScalar),
                Self::AbsMinY => (Self::AbsMin, VectorField::Y),
                Self::AbsMinZ => (Self::AbsMin, VectorField::Z),
                Self::AbsMaxX => (Self::AbsMax, VectorField::XOrScalar),
                Self::AbsMaxY => (Self::AbsMax, VectorField::Y),
                Self::AbsMaxZ => (Self::AbsMax, VectorField::Z),
                Self::OriginX => (Self::Origin, VectorField::XOrScalar),
                Self::OriginY => (Self::Origin, VectorField::Y),
                Self::OriginZ => (Self::Origin, VectorField::Z),
                Self::OldOriginX => (Self::OldOrigin, VectorField::XOrScalar),
                Self::OldOriginY => (Self::OldOrigin, VectorField::Y),
                Self::OldOriginZ => (Self::OldOrigin, VectorField::Z),
                Self::VelocityX => (Self::Velocity, VectorField::XOrScalar),
                Self::VelocityY => (Self::Velocity, VectorField::Y),
                Self::VelocityZ => (Self::Velocity, VectorField::Z),
                Self::AnglesX => (Self::Angles, VectorField::XOrScalar),
                Self::AnglesY => (Self::Angles, VectorField::Y),
                Self::AnglesZ => (Self::Angles, VectorField::Z),
                Self::AngularVelocityX => (Self::AngularVelocity, VectorField::XOrScalar),
                Self::AngularVelocityY => (Self::AngularVelocity, VectorField::Y),
                Self::AngularVelocityZ => (Self::AngularVelocity, VectorField::Z),
                Self::PunchAngleX => (Self::PunchAngle, VectorField::XOrScalar),
                Self::PunchAngleY => (Self::PunchAngle, VectorField::Y),
                Self::PunchAngleZ => (Self::PunchAngle, VectorField::Z),
                Self::MinsX => (Self::Mins, VectorField::XOrScalar),
                Self::MinsY => (Self::Mins, VectorField::Y),
                Self::MinsZ => (Self::Mins, VectorField::Z),
                Self::MaxsX => (Self::Maxs, VectorField::XOrScalar),
                Self::MaxsY => (Self::Maxs, VectorField::Y),
                Self::MaxsZ => (Self::Maxs, VectorField::Z),
                Self::SizeX => (Self::Size, VectorField::XOrScalar),
                Self::SizeY => (Self::Size, VectorField::Y),
                Self::SizeZ => (Self::Size, VectorField::Z),
                Self::ViewOffsetX => (Self::ViewOffset, VectorField::XOrScalar),
                Self::ViewOffsetY => (Self::ViewOffset, VectorField::Y),
                Self::ViewOffsetZ => (Self::ViewOffset, VectorField::Z),
                Self::ViewAngleX => (Self::ViewAngle, VectorField::XOrScalar),
                Self::ViewAngleY => (Self::ViewAngle, VectorField::Y),
                Self::ViewAngleZ => (Self::ViewAngle, VectorField::Z),
                Self::MoveDirectionX => (Self::MoveDirection, VectorField::XOrScalar),
                Self::MoveDirectionY => (Self::MoveDirection, VectorField::Y),
                Self::MoveDirectionZ => (Self::MoveDirection, VectorField::Z),
                other => (*other, VectorField::XOrScalar),
            }
        }

        fn to_u16(&self) -> u16 {
            match self {
                Self::ModelId => 0,
                Self::AbsMin | Self::AbsMinX => 1,
                Self::AbsMinY => 2,
                Self::AbsMinZ => 3,
                Self::AbsMax | Self::AbsMaxX => 4,
                Self::AbsMaxY => 5,
                Self::AbsMaxZ => 6,
                Self::LocalTime => 7,
                Self::MoveType => 8,
                Self::Solid => 9,
                Self::Origin | Self::OriginX => 10,
                Self::OriginY => 11,
                Self::OriginZ => 12,
                Self::OldOrigin | Self::OldOriginX => 13,
                Self::OldOriginY => 14,
                Self::OldOriginZ => 15,
                Self::Velocity | Self::VelocityX => 16,
                Self::VelocityY => 17,
                Self::VelocityZ => 18,
                Self::Angles | Self::AnglesX => 19,
                Self::AnglesY => 20,
                Self::AnglesZ => 21,
                Self::AngularVelocity | Self::AngularVelocityX => 22,
                Self::AngularVelocityY => 23,
                Self::AngularVelocityZ => 24,
                Self::PunchAngle | Self::PunchAngleX => 25,
                Self::PunchAngleY => 26,
                Self::PunchAngleZ => 27,
                Self::ClassName => 28,
                Self::ModelName => 29,
                Self::Frame => 30,
                Self::SkinId => 31,
                Self::Effects => 32,
                Self::Mins | Self::MinsX => 33,
                Self::MinsY => 34,
                Self::MinsZ => 35,
                Self::Maxs | Self::MaxsX => 36,
                Self::MaxsY => 37,
                Self::MaxsZ => 38,
                Self::Size | Self::SizeX => 39,
                Self::SizeY => 40,
                Self::SizeZ => 41,
                Self::OnTouch => 42,
                Self::OnUse => 43,
                Self::OnThink => 44,
                Self::OnBlocked => 45,
                Self::NextThink => 46,
                Self::Ground => 47,
                Self::Health => 48,
                Self::Frags => 49,
                Self::WeaponId => 50,
                Self::WeaponModelName => 51,
                Self::WeaponFrame => 52,
                Self::CurrentAmmo => 53,
                Self::AmmoShells => 54,
                Self::AmmoNails => 55,
                Self::AmmoRockets => 56,
                Self::AmmoCells => 57,
                Self::Items => 58,
                Self::TakeDamage => 59,
                Self::Chain => 60,
                Self::DeadFlag => 61,
                Self::ViewOffset | Self::ViewOffsetX => 62,
                Self::ViewOffsetY => 63,
                Self::ViewOffsetZ => 64,
                Self::Button0 => 65,
                Self::Button1 => 66,
                Self::Button2 => 67,
                Self::Impulse => 68,
                Self::FixAngle => 69,
                Self::ViewAngle | Self::ViewAngleX => 70,
                Self::ViewAngleY => 71,
                Self::ViewAngleZ => 72,
                Self::IdealPitch => 73,
                Self::NetworkName => 74,
                Self::Enemy => 75,
                Self::Flags => 76,
                Self::ColorMapId => 77,
                Self::Team => 78,
                Self::MaxHealth => 79,
                Self::TeleportTime => 80,
                Self::ArmorType => 81,
                Self::ArmorValue => 82,
                Self::WaterLevel => 83,
                Self::WaterType => 84,
                Self::IdealYaw => 85,
                Self::YawSpeed => 86,
                Self::AimEntity => 87,
                Self::GoalEntity => 88,
                Self::SpawnFlags => 89,
                Self::Target => 90,
                Self::TargetName => 91,
                Self::DamageTaken => 92,
                Self::DamageSaved => 93,
                Self::DamageInflictor => 94,
                Self::Owner => 95,
                Self::MoveDirection | Self::MoveDirectionX => 96,
                Self::MoveDirectionY => 97,
                Self::MoveDirectionZ => 98,
                Self::Message => 99,
                Self::Sounds => 100,
                Self::Noise0 => 101,
                Self::Noise1 => 102,
                Self::Noise2 => 103,
                Self::Noise3 => 104,
            }
        }

        fn name(&self) -> &'static str {
            match self {
                Self::ModelId => "modelindex",
                Self::AbsMin => "absmin",
                Self::AbsMinX => "absmin_x",
                Self::AbsMinY => "absmin_y",
                Self::AbsMinZ => "absmin_z",
                Self::AbsMax => "absmax",
                Self::AbsMaxX => "absmax_x",
                Self::AbsMaxY => "absmax_y",
                Self::AbsMaxZ => "absmax_z",
                Self::LocalTime => "ltime",
                Self::MoveType => "movetype",
                Self::Solid => "solid",
                Self::Origin => "origin",
                Self::OriginX => "origin_x",
                Self::OriginY => "origin_y",
                Self::OriginZ => "origin_z",
                Self::OldOrigin => "oldorigin",
                Self::OldOriginX => "oldorigin_x",
                Self::OldOriginY => "oldorigin_y",
                Self::OldOriginZ => "oldorigin_z",
                Self::Velocity => "velocity",
                Self::VelocityX => "velocity_x",
                Self::VelocityY => "velocity_y",
                Self::VelocityZ => "velocity_z",
                Self::Angles => "angles",
                Self::AnglesX => "angles_x",
                Self::AnglesY => "angles_y",
                Self::AnglesZ => "angles_z",
                Self::AngularVelocity => "avelocity",
                Self::AngularVelocityX => "avelocity_x",
                Self::AngularVelocityY => "avelocity_y",
                Self::AngularVelocityZ => "avelocity_z",
                Self::PunchAngle => "punchangle",
                Self::PunchAngleX => "punchangle_x",
                Self::PunchAngleY => "punchangle_y",
                Self::PunchAngleZ => "punchangle_z",
                Self::ClassName => "classname",
                Self::ModelName => "model",
                Self::Frame => "frame",
                Self::SkinId => "skin",
                Self::Effects => "effects",
                Self::Mins => "mins",
                Self::MinsX => "mins_x",
                Self::MinsY => "mins_y",
                Self::MinsZ => "mins_z",
                Self::Maxs => "maxs",
                Self::MaxsX => "maxs_x",
                Self::MaxsY => "maxs_y",
                Self::MaxsZ => "maxs_z",
                Self::Size => "size",
                Self::SizeX => "size_x",
                Self::SizeY => "size_y",
                Self::SizeZ => "size_z",
                Self::OnTouch => "touch",
                Self::OnUse => "use",
                Self::OnThink => "think",
                Self::OnBlocked => "blocked",
                Self::NextThink => "nextthink",
                Self::Ground => "groundentity",
                Self::Health => "health",
                Self::Frags => "frags",
                Self::WeaponId => "weapon",
                Self::WeaponModelName => "weaponmodel",
                Self::WeaponFrame => "weaponframe",
                Self::CurrentAmmo => "currentammo",
                Self::AmmoShells => "ammo_shells",
                Self::AmmoNails => "ammo_nails",
                Self::AmmoRockets => "ammo_rockets",
                Self::AmmoCells => "ammo_cells",
                Self::Items => "items",
                Self::TakeDamage => "takedamage",
                Self::Chain => "chain",
                Self::DeadFlag => "deadflag",
                Self::ViewOffset => "view_ofs",
                Self::ViewOffsetX => "view_ofs_x",
                Self::ViewOffsetY => "view_ofs_y",
                Self::ViewOffsetZ => "view_ofs_z",
                Self::Button0 => "button0",
                Self::Button1 => "button1",
                Self::Button2 => "button2",
                Self::Impulse => "impulse",
                Self::FixAngle => "fixangle",
                Self::ViewAngle => "v_angle",
                Self::ViewAngleX => "v_angle_x",
                Self::ViewAngleY => "v_angle_y",
                Self::ViewAngleZ => "v_angle_z",
                Self::IdealPitch => "idealpitch",
                Self::NetworkName => "netname",
                Self::Enemy => "enemy",
                Self::Flags => "flags",
                Self::ColorMapId => "colormap",
                Self::Team => "team",
                Self::MaxHealth => "max_health",
                Self::TeleportTime => "teleport_time",
                Self::ArmorType => "armortype",
                Self::ArmorValue => "armorvalue",
                Self::WaterLevel => "waterlevel",
                Self::WaterType => "watertype",
                Self::IdealYaw => "ideal_yaw",
                Self::YawSpeed => "yaw_speed",
                Self::AimEntity => "aiment",
                Self::GoalEntity => "goalentity",
                Self::SpawnFlags => "spawnflags",
                Self::Target => "target",
                Self::TargetName => "targetname",
                Self::DamageTaken => "dmg_take",
                Self::DamageSaved => "dmg_save",
                Self::DamageInflictor => "dmg_inflictor",
                Self::Owner => "owner",
                Self::MoveDirection => "movedir",
                Self::MoveDirectionX => "movedir_x",
                Self::MoveDirectionY => "movedir_y",
                Self::MoveDirectionZ => "movedir_z",
                Self::Message => "message",
                Self::Sounds => "sounds",
                Self::Noise0 => "noise",
                Self::Noise1 => "noise1",
                Self::Noise2 => "noise2",
                Self::Noise3 => "noise3",
            }
        }

        fn from_name(name: &str) -> Option<Self> {
            // Can't just iter over `VARIANTS` as that isn't stable in const fns.
            let mut i = 0;
            while i < Self::VARIANTS.len() {
                let variant = Self::VARIANTS[i];

                if variant.name() == name {
                    return Some(variant);
                }

                i += 1;
            }

            None
        }

        fn type_(&self) -> Type {
            match self {
                // float    modelindex;
                Self::ModelId => Type::Float,
                // vec3_t   absmin;
                Self::AbsMin => Type::Vector,
                Self::AbsMinX => Type::Float,
                Self::AbsMinY => Type::Float,
                Self::AbsMinZ => Type::Float,
                // vec3_t   absmax;
                Self::AbsMax => Type::Vector,
                Self::AbsMaxX => Type::Float,
                Self::AbsMaxY => Type::Float,
                Self::AbsMaxZ => Type::Float,
                // float    ltime;
                Self::LocalTime => Type::Float,
                // float    movetype;
                Self::MoveType => Type::Float,
                // float    solid;
                Self::Solid => Type::Float,
                // vec3_t   origin;
                Self::Origin => Type::Vector,
                Self::OriginX => Type::Float,
                Self::OriginY => Type::Float,
                Self::OriginZ => Type::Float,
                // vec3_t   oldorigin;
                Self::OldOrigin => Type::Vector,
                Self::OldOriginX => Type::Float,
                Self::OldOriginY => Type::Float,
                Self::OldOriginZ => Type::Float,
                // vec3_t   velocity;
                Self::Velocity => Type::Vector,
                Self::VelocityX => Type::Float,
                Self::VelocityY => Type::Float,
                Self::VelocityZ => Type::Float,
                // vec3_t   angles;
                Self::Angles => Type::Vector,
                Self::AnglesX => Type::Float,
                Self::AnglesY => Type::Float,
                Self::AnglesZ => Type::Float,
                // vec3_t   avelocity;
                Self::AngularVelocity => Type::Vector,
                Self::AngularVelocityX => Type::Float,
                Self::AngularVelocityY => Type::Float,
                Self::AngularVelocityZ => Type::Float,
                // vec3_t   punchangle;
                Self::PunchAngle => Type::Vector,
                Self::PunchAngleX => Type::Float,
                Self::PunchAngleY => Type::Float,
                Self::PunchAngleZ => Type::Float,
                // string_t classname;
                Self::ClassName => Type::String,
                // string_t model;
                Self::ModelName => Type::String,
                // float    frame;
                Self::Frame => Type::Float,
                // float    skin;
                Self::SkinId => Type::Float,
                // float    effects;
                Self::Effects => Type::Float,
                // vec3_t   mins;
                Self::Mins => Type::Vector,
                Self::MinsX => Type::Float,
                Self::MinsY => Type::Float,
                Self::MinsZ => Type::Float,
                // vec3_t   maxs;
                Self::Maxs => Type::Vector,
                Self::MaxsX => Type::Float,
                Self::MaxsY => Type::Float,
                Self::MaxsZ => Type::Float,
                // vec3_t   size;
                Self::Size => Type::Vector,
                Self::SizeX => Type::Float,
                Self::SizeY => Type::Float,
                Self::SizeZ => Type::Float,
                // func_t   touch;
                Self::OnTouch => Type::Function,
                // func_t   use;
                Self::OnUse => Type::Function,
                // func_t   think;
                Self::OnThink => Type::Function,
                // func_t   blocked;
                Self::OnBlocked => Type::Function,
                // float    nextthink;
                Self::NextThink => Type::Float,
                // int      groundentity;
                Self::Ground => Type::Entity,
                // float    health;
                Self::Health => Type::Float,
                // float    frags;
                Self::Frags => Type::Float,
                // float    weapon;
                Self::WeaponId => Type::Float,
                // string_t weaponmodel;
                Self::WeaponModelName => Type::String,
                // float    weaponframe;
                Self::WeaponFrame => Type::Float,
                // float    currentammo;
                Self::CurrentAmmo => Type::Float,
                // float    ammo_shells;
                Self::AmmoShells => Type::Float,
                // float    ammo_nails;
                Self::AmmoNails => Type::Float,
                // float    ammo_rockets;
                Self::AmmoRockets => Type::Float,
                // float    ammo_cells;
                Self::AmmoCells => Type::Float,
                // float    items;
                Self::Items => Type::Float,
                // float    takedamage;
                Self::TakeDamage => Type::Float,
                // int      chain;
                Self::Chain => Type::Entity,
                // float    deadflag;
                Self::DeadFlag => Type::Float,
                // vec3_t   view_ofs;
                Self::ViewOffset => Type::Vector,
                Self::ViewOffsetX => Type::Float,
                Self::ViewOffsetY => Type::Float,
                Self::ViewOffsetZ => Type::Float,
                // float    button0;
                Self::Button0 => Type::Float,
                // float    button1;
                Self::Button1 => Type::Float,
                // float    button2;
                Self::Button2 => Type::Float,
                // float    impulse;
                Self::Impulse => Type::Float,
                // float    fixangle;
                Self::FixAngle => Type::Float,
                // vec3_t   v_angle;
                Self::ViewAngle => Type::Vector,
                Self::ViewAngleX => Type::Float,
                Self::ViewAngleY => Type::Float,
                Self::ViewAngleZ => Type::Float,
                // float    idealpitch;
                Self::IdealPitch => Type::Float,
                // string_t netname;
                Self::NetworkName => Type::String,
                // int      enemy;
                Self::Enemy => Type::Entity,
                // float    flags;
                Self::Flags => Type::Float,
                // float    colormap;
                Self::ColorMapId => Type::Float,
                // float    team;
                Self::Team => Type::Float,
                // float    max_health;
                Self::MaxHealth => Type::Float,
                // float    teleport_time;
                Self::TeleportTime => Type::Float,
                // float    armortype;
                Self::ArmorType => Type::Float,
                // float    armorvalue;
                Self::ArmorValue => Type::Float,
                // float    waterlevel;
                Self::WaterLevel => Type::Float,
                // float    watertype;
                Self::WaterType => Type::Float,
                // float    ideal_yaw;
                Self::IdealYaw => Type::Float,
                // float    yaw_speed;
                Self::YawSpeed => Type::Float,
                // int      aiment;
                Self::AimEntity => Type::Entity,
                // int      goalentity;
                Self::GoalEntity => Type::Entity,
                // float    spawnflags;
                Self::SpawnFlags => Type::Float,
                // string_t target;
                Self::Target => Type::String,
                // string_t targetname;
                Self::TargetName => Type::String,
                // float    dmg_take;
                Self::DamageTaken => Type::Float,
                // float    dmg_save;
                Self::DamageSaved => Type::Float,
                // int      dmg_inflictor;
                Self::DamageInflictor => Type::Entity,
                // int      owner;
                Self::Owner => Type::Entity,
                // vec3_t   movedir;
                Self::MoveDirection => Type::Vector,
                Self::MoveDirectionX => Type::Float,
                Self::MoveDirectionY => Type::Float,
                Self::MoveDirectionZ => Type::Float,
                // string_t message;
                Self::Message => Type::String,
                // float    sounds;
                Self::Sounds => Type::Float,
                // string_t noise;
                Self::Noise0 => Type::String,
                // string_t noise1;
                Self::Noise1 => Type::String,
                // string_t noise2;
                Self::Noise2 => Type::String,
                // string_t noise3;
                Self::Noise3 => Type::String,
            }
        }
    }
}
