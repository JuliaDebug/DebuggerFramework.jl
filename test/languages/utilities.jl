# Taken from Tokenize.jl
@inline iswhitespace(c::Char) = Base.UTF8proc.isspace(c)

const EOF_CHAR = convert(Char,typemax(UInt32))

function peekchar(io::(isdefined(Base, :GenericIOBuffer) ?
    Base.GenericIOBuffer : Base.AbstractIOBuffer))
    if !io.readable || io.ptr > io.size
        return EOF_CHAR
    end
    ch, _ = readutf(io)
    return ch
end

readchar(io::IO) = eof(io) ? EOF_CHAR : read(io, Char)

function readutf(io, offset = 0)
    ch = convert(UInt8, io.data[io.ptr + offset])
    if ch < 0x80
        return convert(Char, ch), 0
    end
    trailing = Base.utf8_trailing[ch + 1]
    c::UInt32 = 0
    for j = 1:trailing
        c += ch
        c <<= 6
        ch = convert(UInt8, io.data[io.ptr + j + offset])
    end
    c += ch
    c -= Base.utf8_offset[trailing + 1]
    return convert(Char, c), trailing
end

function dpeekchar(io::IOBuffer)
    if !io.readable || io.ptr > io.size
        return EOF_CHAR, EOF_CHAR
    end
    ch1, trailing = readutf(io)
    offset = trailing + 1

    if io.ptr + offset > io.size
        return ch1, EOF_CHAR
    end
    ch2, _ = readutf(io, offset)

    return ch1, ch2
end