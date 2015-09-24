extern "C" {

#ifdef _WIN64
	__int64 __fastcall MandelbrotLineSSE2x64Two(double cr1, double ci1, double cr2, double ci2, int bailout);
	void __fastcall read_timestamp_counter (void *);
#endif
}
