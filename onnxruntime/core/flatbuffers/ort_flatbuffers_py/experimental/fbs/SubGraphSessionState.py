# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SubGraphSessionState(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSubGraphSessionState(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubGraphSessionState()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def SubGraphSessionStateBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # SubGraphSessionState
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubGraphSessionState
    def GraphId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # SubGraphSessionState
    def SessionState(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.experimental.fbs.SessionState import SessionState
            obj = SessionState()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def SubGraphSessionStateStart(builder): builder.StartObject(2)
def SubGraphSessionStateAddGraphId(builder, graphId): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(graphId), 0)
def SubGraphSessionStateAddSessionState(builder, sessionState): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(sessionState), 0)
def SubGraphSessionStateEnd(builder): return builder.EndObject()
