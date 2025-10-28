# `qcvm`

This is a clean-room [QuakeC](https://en.wikipedia.org/wiki/QuakeC) VM implementation,
designed to be integrated into the [Seismon](https://github.com/eira-fransham/seismon)
engine. Unlike most other implementations, it is designed with embedding in mind, and
is not tied to only being used for Quake-like games.

## FAQ

#### Is this the fastest QuakeC VM in the world

It is almost certainly not going to get best-in-class performance for now, as it is
not designed with that in mind - Seismon is built for extensibility and moddability
first and efficiency second, and this VM adopts the same mindset.

#### Why did you make this

Because I want a VM that is resilient enough that hobbyist game developers and modders
can mess around with a repl, override functions, just generally treat the code like
it's a rockstar's hotel room and still have the game engine generally respond in a
reasonable way.

#### Is this a meme

No, I actually have a goal in mind and am taking this project seriously. You could
use this to write a web server in QuakeC, though, and I do think that's very funny.

#### Should I write my game in QuakeC then?

Not unless you want to be sectioned.
