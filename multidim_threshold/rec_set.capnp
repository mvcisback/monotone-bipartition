@0xc7b252b79e0e6255;

struct Rec @0xc561637204d4a6a9 {
     bot @0 :List(UInt32);
     top @1 :List(UInt32);
}


struct RecSet @0x8cb487195391c1ab {
  id @0 :UInt32;
  recs @1 :List(Rec);
}


struct RecSetCollection @0xb0956c36582614ec {
   recSets @0 :List(RecSet);

   # TODO: add metadata here such as cluster id
}