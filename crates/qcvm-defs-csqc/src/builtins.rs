// Generated from FTEQW's and QuakeSpasm-Spiked's CSQC builtin tables; see the crate docs.
//! Builtin numbers of the core CSQC builtins (see [`BuiltinAddr`]).

use std::fmt;

use strum::{EnumIter, VariantArray};

use qcvm::BuiltinAddress;

/// Builtin numbers of the core CSQC builtins: those that both FTEQW and QuakeSpasm-Spiked
/// support in CSQC.
///
/// Builtins that are also Quake 1 builtins (see `qcvm_defs_quake1::BuiltinAddr`) have the same
/// canonical names, so that hosts can register a single implementation for both. Builtins that
/// FTEQW numbers `#0` (i.e. only by name) have no variant, and are looked up by the name in the
/// `progs.dat`.
#[derive(VariantArray, EnumIter, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuiltinAddr {
    /// `#1`: `makevectors` (`void(vector vang)`)
    Makevectors,
    /// `#2`: `setorigin` (`void(entity e, vector o)`)
    Setorigin,
    /// `#3`: `setmodel` (`void(entity e, string m)`)
    Setmodel,
    /// `#4`: `setsize` (`void(entity e, vector min, vector max)`)
    Setsize,
    /// `#6`: `break` (`void()`)
    Break,
    /// `#7`: `random` (`float()`)
    Random,
    /// `#8`: `sound` (`void(entity e, float chan, string samp, float vol, float atten, optional float speedpct, optional float flags, optional float timeofs)`)
    Sound,
    /// `#9`: `normalize` (`vector(vector v)`)
    Normalize,
    /// `#10`: `error` (`void(string e)`)
    Error,
    /// `#11`: `objerror` (`void(string e)`)
    Objerror,
    /// `#12`: `vlen` (`float(vector v)`)
    Vlen,
    /// `#13`: `vectoyaw` (`float(vector v, optional entity reference)`)
    Vectoyaw,
    /// `#14`: `spawn` (`entity()`)
    Spawn,
    /// `#15`: `remove` (`void(entity e)`)
    Remove,
    /// `#16`: `traceline` (`void(vector v1, vector v2, float flags, entity ent)`)
    Traceline,
    /// `#18`: `find` (`entity(entity start, .string fld, string match)`)
    Find,
    /// `#19`: `precache_sound` (`string(string s)`)
    PrecacheSound,
    /// `#20`: `precache_model` (`string(string s)`)
    PrecacheModel,
    /// `#22`: `findradius` (`entity(vector org, float rad, optional .entity chainfield)`)
    Findradius,
    /// `#25`: `dprint` (`void(string s, optional string s2, optional string s3, optional string s4, optional string s5, optional string s6, optional string s7, optional string s8)`)
    Dprint,
    /// `#26`: `ftos` (`string(float val)`)
    Ftos,
    /// `#27`: `vtos` (`string(vector val)`)
    Vtos,
    /// `#28`: `coredump` (`void()`)
    Coredump,
    /// `#29`: `traceon` (`void()`)
    Traceon,
    /// `#30`: `traceoff` (`void()`)
    Traceoff,
    /// `#31`: `eprint` (`void(entity e)`)
    Eprint,
    /// `#32`: `walkmove` (`float(float yaw, float dist, optional float settraceglobals)`)
    Walkmove,
    /// `#34`: `droptofloor` (`float()`)
    Droptofloor,
    /// `#35`: `lightstyle` (`void(float lightstyle, string stylestring, optional vector rgb)`)
    Lightstyle,
    /// `#36`: `rint` (`float(float)`)
    Rint,
    /// `#37`: `floor` (`float(float)`)
    Floor,
    /// `#38`: `ceil` (`float(float)`)
    Ceil,
    /// `#40`: `checkbottom` (`float(entity ent)`)
    Checkbottom,
    /// `#41`: `pointcontents` (`float(vector pos)`)
    Pointcontents,
    /// `#43`: `fabs` (`float(float)`)
    Fabs,
    /// `#45`: `cvar` (`float(string)`)
    Cvar,
    /// `#46`: `localcmd` (`void(string, ...)`)
    Localcmd,
    /// `#47`: `nextent` (`entity(entity)`)
    Nextent,
    /// `#48`: `particle` (`void(vector pos, vector dir, float colour, float count)`)
    Particle,
    /// `#49`: `ChangeYaw` (`void()`)
    ChangeYaw,
    /// `#51`: `vectoangles` (`vector(vector fwd, optional vector up)`)
    Vectoangles,
    /// `#60`: `sin` (`float(float angle)`)
    Sin,
    /// `#61`: `cos` (`float(float angle)`)
    Cos,
    /// `#62`: `sqrt` (`float(float value)`)
    Sqrt,
    /// `#64`: `tracetoss` (`void(entity ent, entity ignore)`)
    Tracetoss,
    /// `#65`: `etos` (`string(entity ent)`)
    Etos,
    /// `#67`: `movetogoal` (`void(float step)`)
    Movetogoal,
    /// `#68`: `precache_file` (`string(string s)`)
    PrecacheFile,
    /// `#69`: `makestatic` (`void(entity e)`)
    Makestatic,
    /// `#72`: `cvar_set` (`void(string cvarname, string valuetoset)`)
    CvarSet,
    /// `#74`: `ambientsound` (`void(vector pos, string samp, float vol, float atten)`)
    Ambientsound,
    /// `#75`: `precache_model2` (`string(string str)`)
    PrecacheModel2,
    /// `#76`: `precache_sound2` (`string(string str)`)
    PrecacheSound2,
    /// `#77`: `precache_file2` (`string(string str)`)
    PrecacheFile2,
    /// `#81`: `stof` (`float(string)`)
    Stof,
    /// `#90`: `tracebox` (`void(vector start, vector mins, vector maxs, vector end, float nomonsters, entity ent)`)
    Tracebox,
    /// `#91`: `randomvec` (`vector()`)
    Randomvec,
    /// `#92`: `getlight` (`vector(vector org)`)
    Getlight,
    /// `#93`: `registercvar` (`float(string cvarname, string defaultvalue, optional float flags)`)
    Registercvar,
    /// `#94`: `min` (`float(float a, float b, ...)`)
    Min,
    /// `#95`: `max` (`float(float a, float b, ...)`)
    Max,
    /// `#96`: `bound` (`float(float minimum, float val, float maximum)`)
    Bound,
    /// `#97`: `pow` (`float(float value, float exp)`)
    Pow,
    /// `#98`: `findfloat` (`entity(entity start, .__variant fld, __variant match)`)
    Findfloat,
    /// `#99`: `checkextension` (`float(string extname)`)
    Checkextension,
    /// `#102`: `anglemod` (`float(float value)`)
    Anglemod,
    /// `#110`: `fopen` (`filestream(string filename, float mode, optional float mmapminsize)`)
    Fopen,
    /// `#111`: `fclose` (`void(filestream fhandle)`)
    Fclose,
    /// `#112`: `fgets` (`string(filestream fhandle)`)
    Fgets,
    /// `#113`: `fputs` (`void(filestream fhandle, string s, optional string s2, optional string s3, optional string s4, optional string s5, optional string s6, optional string s7)`)
    Fputs,
    /// `#114`: `strlen` (`float(string s)`)
    Strlen,
    /// `#115`: `strcat` (`string(string s1, optional string s2, optional string s3, optional string s4, optional string s5, optional string s6, optional string s7, optional string s8)`)
    Strcat,
    /// `#116`: `substring` (`string(string s, float start, float length)`)
    Substring,
    /// `#117`: `stov` (`vector(string s)`)
    Stov,
    /// `#118`: `strzone` (`string(string s, ...)`)
    Strzone,
    /// `#119`: `strunzone` (`void(string s)`)
    Strunzone,
    /// `#177`: `localsound` (`void(string soundname, optional float channel, optional float volume)`)
    Localsound,
    /// `#218`: `bitshift` (`float(float number, float quantity)`)
    Bitshift,
    /// `#221`: `strstrofs` (`float(string s1, string sub, optional float startidx)`)
    Strstrofs,
    /// `#222`: `str2chr` (`float(string str, float index)`)
    Str2chr,
    /// `#223`: `chr2str` (`string(float chr, ...)`)
    Chr2str,
    /// `#224`: `strconv` (`string(float ccase, float redalpha, float redchars, string str, ...)`)
    Strconv,
    /// `#225`: `strpad` (`string(float pad, string str1, ...)`)
    Strpad,
    /// `#226`: `infoadd` (`infostring(infostring old, string key, string value)`)
    Infoadd,
    /// `#227`: `infoget` (`string(infostring info, string key)`)
    Infoget,
    /// `#228`: `strncmp` (`float(string s1, string s2, optional float len, optional float s1ofs, optional float s2ofs)`)
    Strncmp,
    /// `#229`: `strcasecmp` (`float(string s1, string s2)`)
    Strcasecmp,
    /// `#230`: `strncasecmp` (`float(string s1, string s2, float len, optional float s1ofs, optional float s2ofs)`)
    Strncasecmp,
    /// `#240`: `checkpvs` (`float(vector viewpos, entity entity)`)
    Checkpvs,
    /// `#245`: `mod` (`float(float dividend, float divisor)`)
    Mod,
    /// `#259`: `stoi` (`int(string)`)
    Stoi,
    /// `#260`: `itos` (`string(int)`)
    Itos,
    /// `#261`: `stoh` (`int(string)`)
    Stoh,
    /// `#262`: `htos` (`string(int)`)
    Htos,
    /// `#276`: `frameforname` (`float(float modidx, string framename)`)
    Frameforname,
    /// `#277`: `frameduration` (`float(float modidx, float framenum)`)
    Frameduration,
    /// `#279`: `touchtriggers` (`void(optional entity ent, optional vector neworigin)`)
    Touchtriggers,
    /// `#284`: `frametoname` (`string(float modidx, float framenum)`)
    Frametoname,
    /// `#294`: `checkcommand` (`float(string name)`)
    Checkcommand,
    /// `#300`: `clearscene` (`void()`)
    Clearscene,
    /// `#301`: `addentities` (`void(float mask)`)
    Addentities,
    /// `#302`: `addentity` (`void(entity ent)`)
    Addentity,
    /// `#303`: `setproperty` (`float(float property, ...)`)
    Setproperty,
    /// `#304`: `renderscene` (`void()`)
    Renderscene,
    /// `#305`: `dynamiclight_add` (`float(vector org, float radius, vector lightcolours, optional float style, optional string cubemapname, optional float pflags)`)
    DynamiclightAdd,
    /// `#306`: `R_BeginPolygon` (`void(string texturename, optional float flags, optional float is2d)`)
    RBeginPolygon,
    /// `#307`: `R_PolygonVertex` (`void(vector org, vector texcoords, vector rgb, float alpha)`)
    RPolygonVertex,
    /// `#308`: `R_EndPolygon` (`void()`)
    REndPolygon,
    /// `#309`: `getproperty` (`__variant(float property)`)
    Getproperty,
    /// `#310`: `unproject` (`vector(vector v)`)
    Unproject,
    /// `#311`: `project` (`vector(vector v)`)
    Project,
    /// `#316`: `iscachedpic` (`float(string name)`)
    Iscachedpic,
    /// `#317`: `precache_pic` (`string(string name, optional float flags)`)
    PrecachePic,
    /// `#318`: `drawgetimagesize` (`vector(string picname)`)
    Drawgetimagesize,
    /// `#320`: `drawcharacter` (`float(vector position, float character, vector size, vector rgb, float alpha, optional float drawflag)`)
    Drawcharacter,
    /// `#321`: `drawrawstring` (`float(vector position, string text, vector size, vector rgb, float alpha, optional float drawflag)`)
    Drawrawstring,
    /// `#322`: `drawpic` (`float(vector position, string pic, vector size, vector rgb, float alpha, optional float drawflag)`)
    Drawpic,
    /// `#323`: `drawfill` (`float(vector position, vector size, vector rgb, float alpha, optional float drawflag)`)
    Drawfill,
    /// `#324`: `drawsetcliparea` (`void(float x, float y, float width, float height)`)
    Drawsetcliparea,
    /// `#325`: `drawresetcliparea` (`void()`)
    Drawresetcliparea,
    /// `#326`: `drawstring` (`float(vector position, string text, vector size, vector rgb, float alpha, float drawflag)`)
    Drawstring,
    /// `#327`: `stringwidth` (`float(string text, float usecolours, vector fontsize)`)
    Stringwidth,
    /// `#328`: `drawsubpic` (`void(vector pos, vector sz, string pic, vector srcpos, vector srcsz, vector rgb, float alpha, optional float drawflag)`)
    Drawsubpic,
    /// `#330`: `getstati` (`int(float stnum)`)
    Getstati,
    /// `#331`: `getstatf` (`float(float stnum, optional float firstbit, optional float bitcount)`)
    Getstatf,
    /// `#332`: `getstats` (`string(float stnum)`)
    Getstats,
    /// `#333`: `setmodelindex` (`void(entity e, float mdlindex)`)
    Setmodelindex,
    /// `#334`: `modelnameforindex` (`string(float mdlindex)`)
    Modelnameforindex,
    /// `#335`: `particleeffectnum` (`float(string effectname)`)
    Particleeffectnum,
    /// `#336`: `trailparticles` (`void(float effectnum, entity ent, vector start, vector end)`)
    Trailparticles,
    /// `#337`: `pointparticles` (`void(float effectnum, vector origin, optional vector dir, optional float count)`)
    Pointparticles,
    /// `#338`: `cprint` (`void(string s, ...)`)
    Cprint,
    /// `#339`: `print` (`void(string s, ...)`)
    Print,
    /// `#340`: `keynumtostring` (`string(float keynum)`)
    Keynumtostring,
    /// `#341`: `stringtokeynum` (`float(string keyname)`)
    Stringtokeynum,
    /// `#342`: `getkeybind` (`string(float keynum, optional float bindmap, optional float modifier)`)
    Getkeybind,
    /// `#343`: `setcursormode` (`void(float usecursor, optional string cursorimage, optional vector hotspot, optional float scale)`)
    Setcursormode,
    /// `#345`: `getinputstate` (`float(float inputsequencenum)`)
    Getinputstate,
    /// `#346`: `setsensitivityscaler` (`void(float sens)`)
    Setsensitivityscaler,
    /// `#347`: `runstandardplayerphysics` (`void(entity ent)`)
    Runstandardplayerphysics,
    /// `#348`: `getplayerkeyvalue` (`string(float playernum, string keyname)`)
    Getplayerkeyvalue,
    /// `#349`: `isdemo` (`float()`)
    Isdemo,
    /// `#350`: `isserver` (`float()`)
    Isserver,
    /// `#351`: `SetListener` (`void(vector origin, vector forward, vector right, vector up, optional float reverbtype)`)
    SetListener,
    /// `#352`: `registercommand` (`void(string cmdname, optional string desc)`)
    Registercommand,
    /// `#353`: `wasfreed` (`float(entity ent)`)
    Wasfreed,
    /// `#354`: `serverkey` (`string(string key)`)
    Serverkey,
    /// `#355`: `getentitytoken` (`string(optional string resetstring)`)
    Getentitytoken,
    /// `#359`: `sendevent` (`void(string evname, string evargs, ...)`)
    Sendevent,
    /// `#360`: `readbyte` (`float()`)
    Readbyte,
    /// `#361`: `readchar` (`float()`)
    Readchar,
    /// `#362`: `readshort` (`float()`)
    Readshort,
    /// `#363`: `readlong` (`float()`)
    Readlong,
    /// `#364`: `readcoord` (`float()`)
    Readcoord,
    /// `#365`: `readangle` (`float()`)
    Readangle,
    /// `#366`: `readstring` (`string()`)
    Readstring,
    /// `#367`: `readfloat` (`float()`)
    Readfloat,
    /// `#368`: `readentitynum` (`float()`)
    Readentitynum,
    /// `#400`: `copyentity` (`entity(entity from, optional entity to)`)
    Copyentity,
    /// `#402`: `findchain` (`entity(.string field, string match, optional .entity chainfield)`)
    Findchain,
    /// `#403`: `findchainfloat` (`entity(.float fld, float match, optional .entity chainfield)`)
    Findchainfloat,
    /// `#418`: `te_gunshot` (`void(vector org, optional float count)`)
    TeGunshot,
    /// `#419`: `te_spike` (`void(vector org)`)
    TeSpike,
    /// `#420`: `te_superspike` (`void(vector org)`)
    TeSuperspike,
    /// `#421`: `te_explosion` (`void(vector org)`)
    TeExplosion,
    /// `#422`: `te_tarexplosion` (`void(vector org)`)
    TeTarexplosion,
    /// `#423`: `te_wizspike` (`void(vector org)`)
    TeWizspike,
    /// `#424`: `te_knightspike` (`void(vector org)`)
    TeKnightspike,
    /// `#425`: `te_lavasplash` (`void(vector org)`)
    TeLavasplash,
    /// `#426`: `te_teleport` (`void(vector org)`)
    TeTeleport,
    /// `#427`: `te_explosion2` (`void(vector org, float color, float colorlength)`)
    TeExplosion2,
    /// `#428`: `te_lightning1` (`void(entity own, vector start, vector end)`)
    TeLightning1,
    /// `#429`: `te_lightning2` (`void(entity own, vector start, vector end)`)
    TeLightning2,
    /// `#430`: `te_lightning3` (`void(entity own, vector start, vector end)`)
    TeLightning3,
    /// `#431`: `te_beam` (`void(entity own, vector start, vector end)`)
    TeBeam,
    /// `#432`: `vectorvectors` (`void(vector dir)`)
    Vectorvectors,
    /// `#434`: `getsurfacenumpoints` (`float(entity e, float s)`)
    Getsurfacenumpoints,
    /// `#435`: `getsurfacepoint` (`vector(entity e, float s, float n)`)
    Getsurfacepoint,
    /// `#436`: `getsurfacenormal` (`vector(entity e, float s)`)
    Getsurfacenormal,
    /// `#437`: `getsurfacetexture` (`string(entity e, float s)`)
    Getsurfacetexture,
    /// `#438`: `getsurfacenearpoint` (`float(entity e, vector p)`)
    Getsurfacenearpoint,
    /// `#439`: `getsurfaceclippedpoint` (`vector(entity e, float s, vector p)`)
    Getsurfaceclippedpoint,
    /// `#441`: `tokenize` (`float(string s)`)
    Tokenize,
    /// `#442`: `argv` (`string(float n)`)
    Argv,
    /// `#443`: `setattachment` (`void(entity e, entity tagentity, string tagname)`)
    Setattachment,
    /// `#444`: `search_begin` (`searchhandle(string pattern, float flags, float quiet, optional string filterpackage)`)
    SearchBegin,
    /// `#445`: `search_end` (`void(searchhandle handle)`)
    SearchEnd,
    /// `#446`: `search_getsize` (`float(searchhandle handle)`)
    SearchGetsize,
    /// `#447`: `search_getfilename` (`string(searchhandle handle, float num)`)
    SearchGetfilename,
    /// `#448`: `cvar_string` (`string(string cvarname)`)
    CvarString,
    /// `#449`: `findflags` (`entity(entity start, .float fld, float match)`)
    Findflags,
    /// `#450`: `findchainflags` (`entity(.float fld, float match, optional .entity chainfield)`)
    Findchainflags,
    /// `#459`: `edict_num` (`entity(float entnum)`)
    EdictNum,
    /// `#460`: `buf_create` (`strbuf()`)
    BufCreate,
    /// `#461`: `buf_del` (`void(strbuf bufhandle)`)
    BufDel,
    /// `#462`: `buf_getsize` (`float(strbuf bufhandle)`)
    BufGetsize,
    /// `#463`: `buf_copy` (`void(strbuf bufhandle_from, strbuf bufhandle_to)`)
    BufCopy,
    /// `#464`: `buf_sort` (`void(strbuf bufhandle, float sortprefixlen, float backward)`)
    BufSort,
    /// `#465`: `buf_implode` (`string(strbuf bufhandle, string glue)`)
    BufImplode,
    /// `#466`: `bufstr_get` (`string(strbuf bufhandle, float string_index)`)
    BufstrGet,
    /// `#467`: `bufstr_set` (`void(strbuf bufhandle, float string_index, string str)`)
    BufstrSet,
    /// `#468`: `bufstr_add` (`float(strbuf bufhandle, string str, float ordered)`)
    BufstrAdd,
    /// `#469`: `bufstr_free` (`void(strbuf bufhandle, float string_index)`)
    BufstrFree,
    /// `#471`: `asin` (`float(float s)`)
    Asin,
    /// `#472`: `acos` (`float(float c)`)
    Acos,
    /// `#473`: `atan` (`float(float t)`)
    Atan,
    /// `#474`: `atan2` (`float(float c, float s)`)
    Atan2,
    /// `#475`: `tan` (`float(float a)`)
    Tan,
    /// `#476`: `strlennocol` (`float(string s)`)
    Strlennocol,
    /// `#477`: `strdecolorize` (`string(string s)`)
    Strdecolorize,
    /// `#478`: `strftime` (`string(float uselocaltime, string format, ...)`)
    Strftime,
    /// `#479`: `tokenizebyseparator` (`float(string s, string separator1, ...)`)
    Tokenizebyseparator,
    /// `#480`: `strtolower` (`string(string s)`)
    Strtolower,
    /// `#481`: `strtoupper` (`string(string s)`)
    Strtoupper,
    /// `#482`: `cvar_defstring` (`string(string s)`)
    CvarDefstring,
    /// `#483`: `pointsound` (`void(vector origin, string sample, float volume, float attenuation)`)
    Pointsound,
    /// `#484`: `strreplace` (`string(string search, string replace, string subject)`)
    Strreplace,
    /// `#485`: `strireplace` (`string(string search, string replace, string subject)`)
    Strireplace,
    /// `#486`: `getsurfacepointattribute` (`vector(entity e, float s, float n, float a)`)
    Getsurfacepointattribute,
    /// `#494`: `crc16` (`float(float caseinsensitive, string s, ...)`)
    Crc16,
    /// `#495`: `cvar_type` (`float(string name)`)
    CvarType,
    /// `#496`: `numentityfields` (`float()`)
    Numentityfields,
    /// `#497`: `entityfieldname` (`string(float fieldnum)`)
    Entityfieldname,
    /// `#498`: `entityfieldtype` (`float(float fieldnum)`)
    Entityfieldtype,
    /// `#499`: `getentityfieldstring` (`string(float fieldnum, entity ent)`)
    Getentityfieldstring,
    /// `#500`: `putentityfieldstring` (`float(float fieldnum, entity ent, string s)`)
    Putentityfieldstring,
    /// `#503`: `whichpack` (`string(string filename, optional float flags)`)
    Whichpack,
    /// `#504`: `getentity` (`__variant(float entnum, float fieldnum)`)
    Getentity,
    /// `#510`: `uri_escape` (`string(string in)`)
    UriEscape,
    /// `#511`: `uri_unescape` (`string(string in)`)
    UriUnescape,
    /// `#512`: `num_for_edict` (`float(entity ent)`)
    NumForEdict,
    /// `#513`: `uri_get` (`float(string uril, float id, optional string postmimetype, optional string postdata)`)
    UriGet,
    /// `#514`: `tokenize_console` (`float(string str)`)
    TokenizeConsole,
    /// `#515`: `argv_start_index` (`float(float idx)`)
    ArgvStartIndex,
    /// `#516`: `argv_end_index` (`float(float idx)`)
    ArgvEndIndex,
    /// `#517`: `buf_cvarlist` (`void(strbuf strbuf, string pattern, string antipattern)`)
    BufCvarlist,
    /// `#518`: `cvar_description` (`string(string cvarname)`)
    CvarDescription,
    /// `#521`: `findkeysforcommand` (`string(string command, optional float bindmap)`)
    Findkeysforcommand,
    /// `#532`: `log` (`float(float v, optional float base)`)
    Log,
    /// `#534`: `soundlength` (`float(string sample)`)
    Soundlength,
    /// `#535`: `buf_loadfile` (`float(string filename, strbuf bufhandle)`)
    BufLoadfile,
    /// `#536`: `buf_writefile` (`float(filestream filehandle, strbuf bufhandle, optional float startpos, optional float numstrings)`)
    BufWritefile,
    /// `#605`: `callfunction` (`void(...)`)
    Callfunction,
    /// `#606`: `writetofile` (`void(filestream fh, entity e)`)
    Writetofile,
    /// `#607`: `isfunction` (`float(string s)`)
    Isfunction,
    /// `#608`: `getresolution` (`vector(float vidmode, optional float forfullscreen)`)
    Getresolution,
    /// `#611`: `gethostcachevalue` (`float(float type)`)
    Gethostcachevalue,
    /// `#612`: `gethostcachestring` (`string(float fld, float hostnr)`)
    Gethostcachestring,
    /// `#613`: `parseentitydata` (`float(entity e, string s, optional float offset)`)
    Parseentitydata,
    /// `#615`: `resethostcachemasks` (`void()`)
    Resethostcachemasks,
    /// `#616`: `sethostcachemaskstring` (`void(float mask, float fld, string str, float op)`)
    Sethostcachemaskstring,
    /// `#617`: `sethostcachemasknumber` (`void(float mask, float fld, float num, float op)`)
    Sethostcachemasknumber,
    /// `#618`: `resorthostcache` (`void()`)
    Resorthostcache,
    /// `#619`: `sethostcachesort` (`void(float fld, float descending)`)
    Sethostcachesort,
    /// `#620`: `refreshhostcache` (`void(optional float dopurge)`)
    Refreshhostcache,
    /// `#621`: `gethostcachenumber` (`float(float fld, float hostnr)`)
    Gethostcachenumber,
    /// `#622`: `gethostcacheindexforkey` (`float(string key)`)
    Gethostcacheindexforkey,
    /// `#623`: `addwantedhostcachekey` (`void(string key)`)
    Addwantedhostcachekey,
    /// `#626`: `getgamedirinfo` (`string(float n, float prop)`)
    Getgamedirinfo,
    /// `#627`: `sprintf` (`string(string fmt, ...)`)
    Sprintf,
    /// `#628`: `getsurfacenumtriangles` (`float(entity e, float s)`)
    Getsurfacenumtriangles,
    /// `#629`: `getsurfacetriangle` (`vector(entity e, float s, float n)`)
    Getsurfacetriangle,
    /// `#630`: `setkeybind` (`float(float key, string bind, optional float bindmap, optional float modifier)`)
    Setkeybind,
    /// `#631`: `getbindmaps` (`vector()`)
    Getbindmaps,
    /// `#632`: `setbindmaps` (`float(vector bm)`)
    Setbindmaps,
    /// `#639`: `digest_hex` (`string(string digest, string data, ...)`)
    DigestHex,
}

impl BuiltinAddress for BuiltinAddr {
    fn number(&self) -> u32 {
        match self {
            Self::Makevectors => 1,
            Self::Setorigin => 2,
            Self::Setmodel => 3,
            Self::Setsize => 4,
            Self::Break => 6,
            Self::Random => 7,
            Self::Sound => 8,
            Self::Normalize => 9,
            Self::Error => 10,
            Self::Objerror => 11,
            Self::Vlen => 12,
            Self::Vectoyaw => 13,
            Self::Spawn => 14,
            Self::Remove => 15,
            Self::Traceline => 16,
            Self::Find => 18,
            Self::PrecacheSound => 19,
            Self::PrecacheModel => 20,
            Self::Findradius => 22,
            Self::Dprint => 25,
            Self::Ftos => 26,
            Self::Vtos => 27,
            Self::Coredump => 28,
            Self::Traceon => 29,
            Self::Traceoff => 30,
            Self::Eprint => 31,
            Self::Walkmove => 32,
            Self::Droptofloor => 34,
            Self::Lightstyle => 35,
            Self::Rint => 36,
            Self::Floor => 37,
            Self::Ceil => 38,
            Self::Checkbottom => 40,
            Self::Pointcontents => 41,
            Self::Fabs => 43,
            Self::Cvar => 45,
            Self::Localcmd => 46,
            Self::Nextent => 47,
            Self::Particle => 48,
            Self::ChangeYaw => 49,
            Self::Vectoangles => 51,
            Self::Sin => 60,
            Self::Cos => 61,
            Self::Sqrt => 62,
            Self::Tracetoss => 64,
            Self::Etos => 65,
            Self::Movetogoal => 67,
            Self::PrecacheFile => 68,
            Self::Makestatic => 69,
            Self::CvarSet => 72,
            Self::Ambientsound => 74,
            Self::PrecacheModel2 => 75,
            Self::PrecacheSound2 => 76,
            Self::PrecacheFile2 => 77,
            Self::Stof => 81,
            Self::Tracebox => 90,
            Self::Randomvec => 91,
            Self::Getlight => 92,
            Self::Registercvar => 93,
            Self::Min => 94,
            Self::Max => 95,
            Self::Bound => 96,
            Self::Pow => 97,
            Self::Findfloat => 98,
            Self::Checkextension => 99,
            Self::Anglemod => 102,
            Self::Fopen => 110,
            Self::Fclose => 111,
            Self::Fgets => 112,
            Self::Fputs => 113,
            Self::Strlen => 114,
            Self::Strcat => 115,
            Self::Substring => 116,
            Self::Stov => 117,
            Self::Strzone => 118,
            Self::Strunzone => 119,
            Self::Localsound => 177,
            Self::Bitshift => 218,
            Self::Strstrofs => 221,
            Self::Str2chr => 222,
            Self::Chr2str => 223,
            Self::Strconv => 224,
            Self::Strpad => 225,
            Self::Infoadd => 226,
            Self::Infoget => 227,
            Self::Strncmp => 228,
            Self::Strcasecmp => 229,
            Self::Strncasecmp => 230,
            Self::Checkpvs => 240,
            Self::Mod => 245,
            Self::Stoi => 259,
            Self::Itos => 260,
            Self::Stoh => 261,
            Self::Htos => 262,
            Self::Frameforname => 276,
            Self::Frameduration => 277,
            Self::Touchtriggers => 279,
            Self::Frametoname => 284,
            Self::Checkcommand => 294,
            Self::Clearscene => 300,
            Self::Addentities => 301,
            Self::Addentity => 302,
            Self::Setproperty => 303,
            Self::Renderscene => 304,
            Self::DynamiclightAdd => 305,
            Self::RBeginPolygon => 306,
            Self::RPolygonVertex => 307,
            Self::REndPolygon => 308,
            Self::Getproperty => 309,
            Self::Unproject => 310,
            Self::Project => 311,
            Self::Iscachedpic => 316,
            Self::PrecachePic => 317,
            Self::Drawgetimagesize => 318,
            Self::Drawcharacter => 320,
            Self::Drawrawstring => 321,
            Self::Drawpic => 322,
            Self::Drawfill => 323,
            Self::Drawsetcliparea => 324,
            Self::Drawresetcliparea => 325,
            Self::Drawstring => 326,
            Self::Stringwidth => 327,
            Self::Drawsubpic => 328,
            Self::Getstati => 330,
            Self::Getstatf => 331,
            Self::Getstats => 332,
            Self::Setmodelindex => 333,
            Self::Modelnameforindex => 334,
            Self::Particleeffectnum => 335,
            Self::Trailparticles => 336,
            Self::Pointparticles => 337,
            Self::Cprint => 338,
            Self::Print => 339,
            Self::Keynumtostring => 340,
            Self::Stringtokeynum => 341,
            Self::Getkeybind => 342,
            Self::Setcursormode => 343,
            Self::Getinputstate => 345,
            Self::Setsensitivityscaler => 346,
            Self::Runstandardplayerphysics => 347,
            Self::Getplayerkeyvalue => 348,
            Self::Isdemo => 349,
            Self::Isserver => 350,
            Self::SetListener => 351,
            Self::Registercommand => 352,
            Self::Wasfreed => 353,
            Self::Serverkey => 354,
            Self::Getentitytoken => 355,
            Self::Sendevent => 359,
            Self::Readbyte => 360,
            Self::Readchar => 361,
            Self::Readshort => 362,
            Self::Readlong => 363,
            Self::Readcoord => 364,
            Self::Readangle => 365,
            Self::Readstring => 366,
            Self::Readfloat => 367,
            Self::Readentitynum => 368,
            Self::Copyentity => 400,
            Self::Findchain => 402,
            Self::Findchainfloat => 403,
            Self::TeGunshot => 418,
            Self::TeSpike => 419,
            Self::TeSuperspike => 420,
            Self::TeExplosion => 421,
            Self::TeTarexplosion => 422,
            Self::TeWizspike => 423,
            Self::TeKnightspike => 424,
            Self::TeLavasplash => 425,
            Self::TeTeleport => 426,
            Self::TeExplosion2 => 427,
            Self::TeLightning1 => 428,
            Self::TeLightning2 => 429,
            Self::TeLightning3 => 430,
            Self::TeBeam => 431,
            Self::Vectorvectors => 432,
            Self::Getsurfacenumpoints => 434,
            Self::Getsurfacepoint => 435,
            Self::Getsurfacenormal => 436,
            Self::Getsurfacetexture => 437,
            Self::Getsurfacenearpoint => 438,
            Self::Getsurfaceclippedpoint => 439,
            Self::Tokenize => 441,
            Self::Argv => 442,
            Self::Setattachment => 443,
            Self::SearchBegin => 444,
            Self::SearchEnd => 445,
            Self::SearchGetsize => 446,
            Self::SearchGetfilename => 447,
            Self::CvarString => 448,
            Self::Findflags => 449,
            Self::Findchainflags => 450,
            Self::EdictNum => 459,
            Self::BufCreate => 460,
            Self::BufDel => 461,
            Self::BufGetsize => 462,
            Self::BufCopy => 463,
            Self::BufSort => 464,
            Self::BufImplode => 465,
            Self::BufstrGet => 466,
            Self::BufstrSet => 467,
            Self::BufstrAdd => 468,
            Self::BufstrFree => 469,
            Self::Asin => 471,
            Self::Acos => 472,
            Self::Atan => 473,
            Self::Atan2 => 474,
            Self::Tan => 475,
            Self::Strlennocol => 476,
            Self::Strdecolorize => 477,
            Self::Strftime => 478,
            Self::Tokenizebyseparator => 479,
            Self::Strtolower => 480,
            Self::Strtoupper => 481,
            Self::CvarDefstring => 482,
            Self::Pointsound => 483,
            Self::Strreplace => 484,
            Self::Strireplace => 485,
            Self::Getsurfacepointattribute => 486,
            Self::Crc16 => 494,
            Self::CvarType => 495,
            Self::Numentityfields => 496,
            Self::Entityfieldname => 497,
            Self::Entityfieldtype => 498,
            Self::Getentityfieldstring => 499,
            Self::Putentityfieldstring => 500,
            Self::Whichpack => 503,
            Self::Getentity => 504,
            Self::UriEscape => 510,
            Self::UriUnescape => 511,
            Self::NumForEdict => 512,
            Self::UriGet => 513,
            Self::TokenizeConsole => 514,
            Self::ArgvStartIndex => 515,
            Self::ArgvEndIndex => 516,
            Self::BufCvarlist => 517,
            Self::CvarDescription => 518,
            Self::Findkeysforcommand => 521,
            Self::Log => 532,
            Self::Soundlength => 534,
            Self::BufLoadfile => 535,
            Self::BufWritefile => 536,
            Self::Callfunction => 605,
            Self::Writetofile => 606,
            Self::Isfunction => 607,
            Self::Getresolution => 608,
            Self::Gethostcachevalue => 611,
            Self::Gethostcachestring => 612,
            Self::Parseentitydata => 613,
            Self::Resethostcachemasks => 615,
            Self::Sethostcachemaskstring => 616,
            Self::Sethostcachemasknumber => 617,
            Self::Resorthostcache => 618,
            Self::Sethostcachesort => 619,
            Self::Refreshhostcache => 620,
            Self::Gethostcachenumber => 621,
            Self::Gethostcacheindexforkey => 622,
            Self::Addwantedhostcachekey => 623,
            Self::Getgamedirinfo => 626,
            Self::Sprintf => 627,
            Self::Getsurfacenumtriangles => 628,
            Self::Getsurfacetriangle => 629,
            Self::Setkeybind => 630,
            Self::Getbindmaps => 631,
            Self::Setbindmaps => 632,
            Self::DigestHex => 639,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Makevectors => "makevectors",
            Self::Setorigin => "setorigin",
            Self::Setmodel => "setmodel",
            Self::Setsize => "setsize",
            Self::Break => "break",
            Self::Random => "random",
            Self::Sound => "sound",
            Self::Normalize => "normalize",
            Self::Error => "error",
            Self::Objerror => "objerror",
            Self::Vlen => "vlen",
            Self::Vectoyaw => "vectoyaw",
            Self::Spawn => "spawn",
            Self::Remove => "remove",
            Self::Traceline => "traceline",
            Self::Find => "find",
            Self::PrecacheSound => "precache_sound",
            Self::PrecacheModel => "precache_model",
            Self::Findradius => "findradius",
            Self::Dprint => "dprint",
            Self::Ftos => "ftos",
            Self::Vtos => "vtos",
            Self::Coredump => "coredump",
            Self::Traceon => "traceon",
            Self::Traceoff => "traceoff",
            Self::Eprint => "eprint",
            Self::Walkmove => "walkmove",
            Self::Droptofloor => "droptofloor",
            Self::Lightstyle => "lightstyle",
            Self::Rint => "rint",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Checkbottom => "checkbottom",
            Self::Pointcontents => "pointcontents",
            Self::Fabs => "fabs",
            Self::Cvar => "cvar",
            Self::Localcmd => "localcmd",
            Self::Nextent => "nextent",
            Self::Particle => "particle",
            Self::ChangeYaw => "ChangeYaw",
            Self::Vectoangles => "vectoangles",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Sqrt => "sqrt",
            Self::Tracetoss => "tracetoss",
            Self::Etos => "etos",
            Self::Movetogoal => "movetogoal",
            Self::PrecacheFile => "precache_file",
            Self::Makestatic => "makestatic",
            Self::CvarSet => "cvar_set",
            Self::Ambientsound => "ambientsound",
            Self::PrecacheModel2 => "precache_model2",
            Self::PrecacheSound2 => "precache_sound2",
            Self::PrecacheFile2 => "precache_file2",
            Self::Stof => "stof",
            Self::Tracebox => "tracebox",
            Self::Randomvec => "randomvec",
            Self::Getlight => "getlight",
            Self::Registercvar => "registercvar",
            Self::Min => "min",
            Self::Max => "max",
            Self::Bound => "bound",
            Self::Pow => "pow",
            Self::Findfloat => "findfloat",
            Self::Checkextension => "checkextension",
            Self::Anglemod => "anglemod",
            Self::Fopen => "fopen",
            Self::Fclose => "fclose",
            Self::Fgets => "fgets",
            Self::Fputs => "fputs",
            Self::Strlen => "strlen",
            Self::Strcat => "strcat",
            Self::Substring => "substring",
            Self::Stov => "stov",
            Self::Strzone => "strzone",
            Self::Strunzone => "strunzone",
            Self::Localsound => "localsound",
            Self::Bitshift => "bitshift",
            Self::Strstrofs => "strstrofs",
            Self::Str2chr => "str2chr",
            Self::Chr2str => "chr2str",
            Self::Strconv => "strconv",
            Self::Strpad => "strpad",
            Self::Infoadd => "infoadd",
            Self::Infoget => "infoget",
            Self::Strncmp => "strncmp",
            Self::Strcasecmp => "strcasecmp",
            Self::Strncasecmp => "strncasecmp",
            Self::Checkpvs => "checkpvs",
            Self::Mod => "mod",
            Self::Stoi => "stoi",
            Self::Itos => "itos",
            Self::Stoh => "stoh",
            Self::Htos => "htos",
            Self::Frameforname => "frameforname",
            Self::Frameduration => "frameduration",
            Self::Touchtriggers => "touchtriggers",
            Self::Frametoname => "frametoname",
            Self::Checkcommand => "checkcommand",
            Self::Clearscene => "clearscene",
            Self::Addentities => "addentities",
            Self::Addentity => "addentity",
            Self::Setproperty => "setproperty",
            Self::Renderscene => "renderscene",
            Self::DynamiclightAdd => "dynamiclight_add",
            Self::RBeginPolygon => "R_BeginPolygon",
            Self::RPolygonVertex => "R_PolygonVertex",
            Self::REndPolygon => "R_EndPolygon",
            Self::Getproperty => "getproperty",
            Self::Unproject => "unproject",
            Self::Project => "project",
            Self::Iscachedpic => "iscachedpic",
            Self::PrecachePic => "precache_pic",
            Self::Drawgetimagesize => "drawgetimagesize",
            Self::Drawcharacter => "drawcharacter",
            Self::Drawrawstring => "drawrawstring",
            Self::Drawpic => "drawpic",
            Self::Drawfill => "drawfill",
            Self::Drawsetcliparea => "drawsetcliparea",
            Self::Drawresetcliparea => "drawresetcliparea",
            Self::Drawstring => "drawstring",
            Self::Stringwidth => "stringwidth",
            Self::Drawsubpic => "drawsubpic",
            Self::Getstati => "getstati",
            Self::Getstatf => "getstatf",
            Self::Getstats => "getstats",
            Self::Setmodelindex => "setmodelindex",
            Self::Modelnameforindex => "modelnameforindex",
            Self::Particleeffectnum => "particleeffectnum",
            Self::Trailparticles => "trailparticles",
            Self::Pointparticles => "pointparticles",
            Self::Cprint => "cprint",
            Self::Print => "print",
            Self::Keynumtostring => "keynumtostring",
            Self::Stringtokeynum => "stringtokeynum",
            Self::Getkeybind => "getkeybind",
            Self::Setcursormode => "setcursormode",
            Self::Getinputstate => "getinputstate",
            Self::Setsensitivityscaler => "setsensitivityscaler",
            Self::Runstandardplayerphysics => "runstandardplayerphysics",
            Self::Getplayerkeyvalue => "getplayerkeyvalue",
            Self::Isdemo => "isdemo",
            Self::Isserver => "isserver",
            Self::SetListener => "SetListener",
            Self::Registercommand => "registercommand",
            Self::Wasfreed => "wasfreed",
            Self::Serverkey => "serverkey",
            Self::Getentitytoken => "getentitytoken",
            Self::Sendevent => "sendevent",
            Self::Readbyte => "readbyte",
            Self::Readchar => "readchar",
            Self::Readshort => "readshort",
            Self::Readlong => "readlong",
            Self::Readcoord => "readcoord",
            Self::Readangle => "readangle",
            Self::Readstring => "readstring",
            Self::Readfloat => "readfloat",
            Self::Readentitynum => "readentitynum",
            Self::Copyentity => "copyentity",
            Self::Findchain => "findchain",
            Self::Findchainfloat => "findchainfloat",
            Self::TeGunshot => "te_gunshot",
            Self::TeSpike => "te_spike",
            Self::TeSuperspike => "te_superspike",
            Self::TeExplosion => "te_explosion",
            Self::TeTarexplosion => "te_tarexplosion",
            Self::TeWizspike => "te_wizspike",
            Self::TeKnightspike => "te_knightspike",
            Self::TeLavasplash => "te_lavasplash",
            Self::TeTeleport => "te_teleport",
            Self::TeExplosion2 => "te_explosion2",
            Self::TeLightning1 => "te_lightning1",
            Self::TeLightning2 => "te_lightning2",
            Self::TeLightning3 => "te_lightning3",
            Self::TeBeam => "te_beam",
            Self::Vectorvectors => "vectorvectors",
            Self::Getsurfacenumpoints => "getsurfacenumpoints",
            Self::Getsurfacepoint => "getsurfacepoint",
            Self::Getsurfacenormal => "getsurfacenormal",
            Self::Getsurfacetexture => "getsurfacetexture",
            Self::Getsurfacenearpoint => "getsurfacenearpoint",
            Self::Getsurfaceclippedpoint => "getsurfaceclippedpoint",
            Self::Tokenize => "tokenize",
            Self::Argv => "argv",
            Self::Setattachment => "setattachment",
            Self::SearchBegin => "search_begin",
            Self::SearchEnd => "search_end",
            Self::SearchGetsize => "search_getsize",
            Self::SearchGetfilename => "search_getfilename",
            Self::CvarString => "cvar_string",
            Self::Findflags => "findflags",
            Self::Findchainflags => "findchainflags",
            Self::EdictNum => "edict_num",
            Self::BufCreate => "buf_create",
            Self::BufDel => "buf_del",
            Self::BufGetsize => "buf_getsize",
            Self::BufCopy => "buf_copy",
            Self::BufSort => "buf_sort",
            Self::BufImplode => "buf_implode",
            Self::BufstrGet => "bufstr_get",
            Self::BufstrSet => "bufstr_set",
            Self::BufstrAdd => "bufstr_add",
            Self::BufstrFree => "bufstr_free",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Atan2 => "atan2",
            Self::Tan => "tan",
            Self::Strlennocol => "strlennocol",
            Self::Strdecolorize => "strdecolorize",
            Self::Strftime => "strftime",
            Self::Tokenizebyseparator => "tokenizebyseparator",
            Self::Strtolower => "strtolower",
            Self::Strtoupper => "strtoupper",
            Self::CvarDefstring => "cvar_defstring",
            Self::Pointsound => "pointsound",
            Self::Strreplace => "strreplace",
            Self::Strireplace => "strireplace",
            Self::Getsurfacepointattribute => "getsurfacepointattribute",
            Self::Crc16 => "crc16",
            Self::CvarType => "cvar_type",
            Self::Numentityfields => "numentityfields",
            Self::Entityfieldname => "entityfieldname",
            Self::Entityfieldtype => "entityfieldtype",
            Self::Getentityfieldstring => "getentityfieldstring",
            Self::Putentityfieldstring => "putentityfieldstring",
            Self::Whichpack => "whichpack",
            Self::Getentity => "getentity",
            Self::UriEscape => "uri_escape",
            Self::UriUnescape => "uri_unescape",
            Self::NumForEdict => "num_for_edict",
            Self::UriGet => "uri_get",
            Self::TokenizeConsole => "tokenize_console",
            Self::ArgvStartIndex => "argv_start_index",
            Self::ArgvEndIndex => "argv_end_index",
            Self::BufCvarlist => "buf_cvarlist",
            Self::CvarDescription => "cvar_description",
            Self::Findkeysforcommand => "findkeysforcommand",
            Self::Log => "log",
            Self::Soundlength => "soundlength",
            Self::BufLoadfile => "buf_loadfile",
            Self::BufWritefile => "buf_writefile",
            Self::Callfunction => "callfunction",
            Self::Writetofile => "writetofile",
            Self::Isfunction => "isfunction",
            Self::Getresolution => "getresolution",
            Self::Gethostcachevalue => "gethostcachevalue",
            Self::Gethostcachestring => "gethostcachestring",
            Self::Parseentitydata => "parseentitydata",
            Self::Resethostcachemasks => "resethostcachemasks",
            Self::Sethostcachemaskstring => "sethostcachemaskstring",
            Self::Sethostcachemasknumber => "sethostcachemasknumber",
            Self::Resorthostcache => "resorthostcache",
            Self::Sethostcachesort => "sethostcachesort",
            Self::Refreshhostcache => "refreshhostcache",
            Self::Gethostcachenumber => "gethostcachenumber",
            Self::Gethostcacheindexforkey => "gethostcacheindexforkey",
            Self::Addwantedhostcachekey => "addwantedhostcachekey",
            Self::Getgamedirinfo => "getgamedirinfo",
            Self::Sprintf => "sprintf",
            Self::Getsurfacenumtriangles => "getsurfacenumtriangles",
            Self::Getsurfacetriangle => "getsurfacetriangle",
            Self::Setkeybind => "setkeybind",
            Self::Getbindmaps => "getbindmaps",
            Self::Setbindmaps => "setbindmaps",
            Self::DigestHex => "digest_hex",
        }
    }
}

impl fmt::Display for BuiltinAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The core CSQC builtins that FTEQW numbers `#0`, i.e. that are only known by name.
pub const NAMED_BUILTINS: &[&str] = &[
    "checkbuiltin",
    "crossproduct",
    "entityfieldref",
    "findentityfield",
    "findkeysforcommandex",
    "fseek",
    "ftoi",
    "ftou",
    "getcursormode",
    "getplayerkeyfloat",
    "gettimed",
    "itof",
    "readdouble",
    "readint",
    "readint64",
    "readuint64",
    "search_getfilemtime",
    "search_getfilesize",
    "search_getpackagename",
    "serverkeyfloat",
    "setwindowcaption",
    "strtrim",
    "utof",
];
