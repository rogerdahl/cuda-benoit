#include "pch.h"

//#include <windows.h>
//#include <malloc.h>    
//#include <stdio.h>
//#include <tchar.h>

//typedef BOOL (WINAPI *LPFN_GLPI)(
//	PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, 
//	PDWORD);

// Get number of active processor cores.
int get_core_count() {
	return 8;
	//BOOL done;
	//BOOL rc;
	//DWORD returnLength;
	//DWORD procCoreCount;
	//DWORD byteOffset;
	//PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer, ptr;
	//LPFN_GLPI Glpi;
	//
	//Glpi = (LPFN_GLPI) GetProcAddress(
	//	GetModuleHandle(TEXT("kernel32")),
	//	"GetLogicalProcessorInformation");
	//
	//if (NULL == Glpi)  {
	//	// GetLogicalProcessorInformation is not supported.
	//	return 0;
	//}
	//
	//done = FALSE;
	//buffer = NULL;
	//returnLength = 0;
	//
	//while (!done) {
	//	rc = Glpi(buffer, &returnLength);
	//
	//	if (FALSE == rc)  {
	//		if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
	//			if (buffer) 
	//				free(buffer);
	//
	//			buffer=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
	//				returnLength);
	//
	//			if (NULL == buffer) {
	//				// Allocation failure
	//				return 0;
	//			}
	//		} 
	//		else  {
	//			// Error: GetLastError()
	//			return 0;
	//		}
	//	} 
	//	else done = TRUE;
	//}
	//
	//procCoreCount = 0;
	//byteOffset = 0;
	//
	//ptr=buffer;
	//while (byteOffset < returnLength) {
	//	switch (ptr->Relationship) {
	//	case RelationProcessorCore:
	//		procCoreCount++;
	//		break;
	//
	//	default:
	//		break;
	//	}
	//	byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
	//	ptr++;
	//}
	//
	//_tprintf(TEXT(": %d\n"), 
	//	procCoreCount);
	//free (buffer);
	//
	//return procCoreCount;
}
