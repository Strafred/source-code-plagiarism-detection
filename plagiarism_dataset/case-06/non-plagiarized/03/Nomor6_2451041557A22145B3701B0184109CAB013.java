import java.util.Scanner;
public class Nomor6_2451041557A22145B3701B0184109CAB013 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int Array[] = new int[10];
        for(int i = 0 ; i<10;i++){
            System.out.print("Read a number: ");
            Array[i] = sc.nextInt();
        }
        for(int i = 9 ; i >= 0 ; i--){
            System.out.println(Array[i]);
        }
    }
}
