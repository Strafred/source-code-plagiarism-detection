import java.util.Scanner;
public class Kasus2L4 {
    public static void main(String[] args) {
        System.out.print("Enter the radius and length of a cylinder: ");
        Scanner in = new Scanner(System.in);
        float radius = in.nextFloat();
        float length = in.nextFloat();
        double luas = hitungLuas(radius);
        System.out.println("The area is " + luas);
        double volume = hitungVolume(luas, length);
        System.out.println("The volume of the cylinder is " + volume);
    }
    private static double hitungLuas(float radius){
        return radius * radius * 3.14159;
    }
    private static double hitungVolume(double luas, float length){
        return luas * length;
    }
}
